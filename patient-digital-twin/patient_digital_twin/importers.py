# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importers: turn segmentations, model output, or mesh files into an ``AnatomyCollection``.

Every importer is constructed with its inputs and returns anatomy from
``to_anatomy_collection()``; wrap the result in ``HumanBody``.

- ``SegmentationImporter``: an existing labeled NIfTI + label dictionary, a directory
  of binary NIfTI masks (one file per structure), or a ZYX NumPy label array. Meshes
  each present label (marching cubes + the image's full affine), centers it in a local
  frame, and records the source label volume and registration on the collection.
- ``NVSegmentImporter``: runs the NV-Segment-CTMR MONAI bundle on a CT/MR NIfTI and
  meshes the requested catalog labels from its output.
- ``NVGenerateImporter``: runs NV-Generate-CTMR to synthesize a paired CT and label
  volume; meshes the labels and keeps the CT on ``ct_scan`` for ``HumanBody.attach_scan``.
- ``SimpleImporter``: named STL/OBJ meshes (XYZ meters) with optional rigid placements.
- ``segmentation_anatomy``: mesh a backend label NIfTI against the catalog (used by the
  NV importers, and by tests that mock inference).

Model backends are optional and imported only when inference runs. Upstream code uses
paths relative to its checkout, so in-process inference temporarily changes the working
directory under a lock; pass ``python_executable`` to run it in a separate process.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any, cast

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage
from numpy.typing import ArrayLike, NDArray

from .anatomy import CATALOG, AnatomyCollection, canonical_name
from .geometry import (
    mask_to_mesh,
    rigid_transform,
    transform_points,
    validate_triangles,
)
from .scan_volume import ScanVolume

SKIP = re.compile(r"^(background|body|dummy\d*|.*_trunc)$")
_UNITS = {"unknown": 0.001, "mm": 0.001, "meter": 1.0, "micron": 1e-6}


def _load_nifti(path: str | Path) -> SpatialImage:
    """``nib.load`` typed as the spatial image every caller expects."""
    return cast(SpatialImage, nib.load(str(path)))


def nifti_affine_m(image: SpatialImage) -> NDArray[np.float64]:
    """Return a 3D NIfTI's XYZ-voxel-to-world 4x4 affine in meters (missing units mean mm).

    Example:
        ``nifti_affine_m(nib.load("labels.nii.gz"))``
    """
    header: Any = image.header  # NIfTI headers carry spatial units; the stub type does not.
    unit = header.get_xyzt_units()[0]
    if len(image.shape) != 3 or unit not in _UNITS:
        raise ValueError(f"Expected a 3D NIfTI in supported units, got {image.shape} in {unit}")
    affine = image.affine.copy()
    affine[:3] *= _UNITS[unit]
    return affine


def _labelmap(value: Mapping[Any, Any] | str | Path) -> dict[int, str]:
    """Normalize an ID->name or NV-Generate name->ID mapping (or a JSON file of either) to ID->name."""
    mapping: Mapping[Any, Any] = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    try:
        items = [(int(k), v) if isinstance(v, str) else (int(v), k) for k, v in dict(mapping).items()]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid labelmap: {exc}") from exc
    result = dict(items)
    if not result or len(result) != len(items) or min(result) < 0 or not all(
        isinstance(name, str) and name.strip() for name in result.values()
    ):
        raise ValueError("labelmap must be a non-empty, unique ID<->name mapping")
    return result


Region = tuple[NDArray[np.intp], NDArray[np.bool_]]


def _crop(mask: NDArray[np.bool_]) -> Region | None:
    """``(low_zyx, cropped mask)`` for a binary ZYX mask, or None when it is empty."""
    from scipy.ndimage import find_objects

    box = find_objects(mask.astype(np.uint8))
    if not box or box[0] is None:
        return None
    low = np.array([s.start for s in box[0]])
    return low, mask[box[0]].copy()


def _union(first: Region | None, second: Region | None) -> Region | None:
    """Union two cropped masks on the same grid (for files that share a canonical name)."""
    if first is None or second is None:
        return first or second
    low = np.minimum(first[0], second[0])
    high = np.maximum(first[0] + first[1].shape, second[0] + second[1].shape)
    window = np.zeros(high - low, bool)
    for start, mask in (first, second):
        offset = start - low
        window[tuple(slice(o, o + n) for o, n in zip(offset, mask.shape))] |= mask
    return low, window


def _load_masks(
    path: Path, names: Iterable[str] | None
) -> tuple[tuple[int, int, int], dict[str, Region | None], NDArray[np.float64]]:
    """Read a directory of binary NIfTI masks (named by file) as independent cropped masks.

    Masks may overlap: each structure keeps its own voxels. Returns ``(grid shape ZYX,
    {canonical name: (low_zyx, cropped mask) or None when empty}, voxel-to-meter affine)``.
    """
    files = {f.name.removesuffix(".gz").removesuffix(".nii"): f
             for f in sorted(path.glob("*.nii")) + sorted(path.glob("*.nii.gz"))}
    files = {n: f for n, f in files.items() if not SKIP.fullmatch(canonical_name(n))}
    if names is not None:
        selected = {canonical_name(n) for n in names}
        if missing := selected - {canonical_name(n) for n in files}:
            raise ValueError(f"Missing binary masks: {sorted(missing)}")
        files = {n: f for n, f in files.items() if canonical_name(n) in selected}
    if not files:
        raise FileNotFoundError(f"No NIfTI masks in {path}")
    reference = _load_nifti(next(iter(files.values())))
    shape, affine = reference.shape[::-1], nifti_affine_m(reference)
    regions: dict[str, Region | None] = {}
    for name, file in files.items():
        image = _load_nifti(file)
        data = np.asanyarray(image.dataobj)
        if image.shape != shape[::-1] or not np.allclose(nifti_affine_m(image), affine, atol=1e-8):
            raise ValueError(f"Mask grid/affine mismatch: {file}")
        if not np.isin(data, [0, 1]).all():
            raise ValueError(f"Expected a binary mask: {file}")
        key = canonical_name(name)
        region = _crop(data.transpose(2, 1, 0).astype(bool))
        regions[key] = _union(regions[key], region) if key in regions else region
    return (int(shape[0]), int(shape[1]), int(shape[2])), regions, affine


class SegmentationImporter:
    """Mesh an existing segmentation: a labeled NIfTI, a binary-mask directory, or a label array.

    Construction reads and validates the labels into ``masks``: one cropped boolean mask per
    canonical structure name (``(low_zyx, mask)``, or None when empty) on the shared
    ``shape_zyx`` grid. Directory masks may overlap; each structure keeps its own voxels.
    ``to_anatomy_collection()`` meshes every mask independently.

    Examples:
        ``SegmentationImporter("labels.nii.gz", "label_dict.json")`` (NV-Generate name->ID JSON),
        ``SegmentationImporter("labels.nii.gz", {1: "liver", 5: "aorta"})``,
        ``SegmentationImporter("masks/", names=["aorta"])`` (one ``<name>.nii.gz`` per structure),
        ``SegmentationImporter(labels_zyx, {1: "liver"}, affine_xyz_to_imaging_m=affine)``.
    """

    def __init__(
        self,
        source: str | Path | ArrayLike,
        labelmap: Mapping[Any, Any] | str | Path | None = None,
        *,
        names: Iterable[str] | None = None,
        affine_xyz_to_imaging_m: ArrayLike | None = None,
    ) -> None:
        """Read labels from ``source``.

        Args:
            source: Labeled NIfTI path, binary-mask directory, or ZYX integer array.
            labelmap: ID->name or name->ID mapping or JSON path; required unless ``source``
                is a directory (whose filenames are the names).
            names: Directory only: load just these structures.
            affine_xyz_to_imaging_m: Array only: XYZ voxel index to meters (default 1 mm voxels).
        """
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            self.shape_zyx, self.masks, affine = _load_masks(Path(source), names)
            self.affine_xyz_to_imaging_m = affine.copy()
            return
        if names is not None:
            raise ValueError("names selects files in a binary-mask directory only")
        if labelmap is None:
            raise ValueError("Pass the matching label dictionary (e.g. configs/label_dict.json)")
        labels, source_affine = _labelmap(labelmap), affine_xyz_to_imaging_m
        source_volume: ArrayLike
        if isinstance(source, (str, Path)):
            image = _load_nifti(source)
            source_volume, source_affine = np.asanyarray(image.dataobj).transpose(2, 1, 0), nifti_affine_m(image)
        else:
            source_volume = source
        volume = np.asarray(source_volume)
        affine = np.diag([0.001] * 3 + [1.0]) if source_affine is None else np.asarray(source_affine, dtype=float)
        if (volume.ndim != 3 or min(volume.shape) == 0 or not np.isfinite(volume).all()
                or np.any(volume < 0) or np.any(volume != np.floor(volume))
                or np.any(volume > np.iinfo(np.int32).max)):
            raise ValueError("Expected a finite non-negative 3D integer segmentation")
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-18):
            raise ValueError("Expected an invertible finite voxel-to-imaging affine")
        if missing := set(map(int, np.unique(volume))) - {0} - labels.keys():
            raise ValueError(f"Present IDs missing from labelmap: {sorted(missing)}")
        self.affine_xyz_to_imaging_m = affine.copy()
        self.shape_zyx = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
        self.masks = self._label_regions(volume.astype(np.int32), labels)

    @staticmethod
    def _label_regions(volume: NDArray[np.int32], labels: Mapping[int, str]) -> dict[str, Region | None]:
        """Split a label volume into one cropped mask per canonical name (IDs sharing a name merge)."""
        from scipy.ndimage import find_objects

        boxes = find_objects(volume)
        ids_by_name: dict[str, list[int]] = {}
        for label_id, raw in labels.items():
            name = canonical_name(raw)
            if label_id and not SKIP.fullmatch(name):
                ids_by_name.setdefault(name, []).append(label_id)
        regions: dict[str, Region | None] = {}
        for name, ids in ids_by_name.items():
            present = [i for i in ids if i <= len(boxes) and boxes[i - 1] is not None]
            if not present:
                regions[name] = None
                continue
            low = np.min([[b.start for b in boxes[i - 1]] for i in present], axis=0)
            high = np.max([[b.stop for b in boxes[i - 1]] for i in present], axis=0)
            regions[name] = low, np.isin(volume[tuple(slice(a, b) for a, b in zip(low, high))], present)
        return regions

    def to_anatomy_collection(self, *, strict: bool = False) -> AnatomyCollection:
        """Mesh every present label into a structure; labels with no voxels stay empty.

        Each mesh is centered in its own local frame (XYZ meters) with a rigid
        ``local_to_body``; the body origin is the anatomy's bounding-box center, and
        ``body_to_imaging`` maps it back to the image's RAS meters. With ``strict=True``
        label names missing from the catalog raise ``ValueError``.
        """
        body = AnatomyCollection.from_names(self.masks, strict=strict)
        bounds: list[NDArray[np.floating]] = []
        flip = np.linalg.det(self.affine_xyz_to_imaging_m[:3, :3]) < 0
        for name, region in self.masks.items():
            if region is None:
                continue
            low, window = region
            vertices, faces = mask_to_mesh(window)  # Voxel XYZ; the full affine follows.
            vertices = transform_points(vertices + low[::-1], self.affine_xyz_to_imaging_m)
            structure, center = body.structures[name], vertices.mean(0)
            # A reflected affine must reverse winding to keep normals outward.
            structure.vertices, structure.faces = vertices - center, faces[:, ::-1].copy() if flip else faces
            structure.local_to_body[:3, 3] = center
            bounds += [vertices.min(0), vertices.max(0)]
        registration = np.eye(4)
        registration[:3, 3] = (np.min(bounds, 0) + np.max(bounds, 0)) / 2 if bounds else 0
        body.body_to_imaging = registration
        for structure in body.structures.values():
            if structure.mesh.vertices is not None:
                structure.local_to_body[:3, 3] -= registration[:3, 3]
                structure.local_to_world = structure.local_to_body.copy()
        return body


def _supported(labelmap: Mapping[Any, str], names: Iterable[str] | None = None) -> dict[int, str]:
    """Backend label IDs whose names are in the catalog (and in names, when given)."""
    labels = {int(i): canonical_name(n) for i, n in labelmap.items()}
    labels = {i: n for i, n in labels.items() if n in CATALOG}
    wanted = None if names is None else {canonical_name(n) for n in names}
    if wanted is not None and (not wanted or wanted - set(labels.values())):
        raise ValueError(f"Unsupported anatomical classes: {sorted(wanted - set(labels.values())) or 'none'}")
    return {i: n for i, n in labels.items() if wanted is None or n in wanted}


def segmentation_anatomy(
    image: SpatialImage, labelmap: Mapping[Any, str], *, names: Iterable[str] | None = None
) -> AnatomyCollection:
    """Mesh a backend's label NIfTI, keeping only catalog labels (and only ``names``, if given).

    Example:
        ``segmentation_anatomy(nib.load("seg.nii.gz"), {6: "aorta"}, names=["aorta"])``
    """
    labels = _supported(labelmap, names)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or np.any(data != np.floor(data)):
        raise ValueError("Backend output must be a finite integer-valued 3D segmentation")
    data = np.where(np.isin(data, list(labels)), data, 0).transpose(2, 1, 0)
    importer = SegmentationImporter(data, labels, affine_xyz_to_imaging_m=nifti_affine_m(image))
    return importer.to_anatomy_collection()


_BACKEND_LOCK = RLock()


@contextmanager
def _backend(root: Path) -> Iterator[None]:
    """Serialize in-process inference; upstream needs its checkout as cwd and `scripts` package."""
    def scripts() -> list[str]:
        """Names of loaded modules in the upstream top-level ``scripts`` package."""
        return [n for n in sys.modules if n == "scripts" or n.startswith("scripts.")]

    with _BACKEND_LOCK:
        cwd, path, argv = Path.cwd(), sys.path[:], sys.argv
        saved = {name: sys.modules.pop(name) for name in scripts()}
        try:
            sys.path.insert(0, str(root))
            os.chdir(root)
            importlib.invalidate_caches()
            yield
        finally:
            for name in scripts():
                del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:], sys.argv = path, argv
            os.chdir(cwd)
            importlib.invalidate_caches()


class NVSegmentImporter:
    """Segment a CT or MR NIfTI with the NV-Segment-CTMR MONAI bundle, then mesh the result.

    Install ``patient-digital-twin[nvsegment]`` and clone the upstream bundle with weights.

    Example:
        ``NVSegmentImporter("ct.nii.gz", bundle_root="NV-Segment-CTMR/NV-Segment-CTMR")``
        ``.to_anatomy_collection(names=["aorta"])``
    """

    def __init__(
        self,
        image: str | Path | SpatialImage,
        *,
        bundle_root: str | Path | None = None,
        modality: str = "CT",
        python_executable: str | Path | None = None,
    ) -> None:
        """Prepare the input image (stored as a millimeter NIfTI on ``self.image``).

        Args:
            image: 3D NIfTI path or loaded image.
            bundle_root: Inner NV-Segment-CTMR bundle directory (default ``$NV_SEGMENT_CTMR_ROOT``).
            modality: ``"CT"`` or ``"MR"``.
            python_executable: Run inference in this interpreter instead of in-process.
        """
        loaded = _load_nifti(image) if isinstance(image, (str, Path)) else image
        data = np.asanyarray(loaded.dataobj)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError("Input image must be a finite 3D volume")
        affine = nifti_affine_m(loaded)
        affine[:3] *= 1000  # Backends assume NIfTI millimeters regardless of header units.
        self.image = nib.Nifti1Image(data.astype(np.float32), affine)
        self.image.header.set_xyzt_units("mm")
        self.root = Path(bundle_root or os.environ.get("NV_SEGMENT_CTMR_ROOT", ".")).resolve()
        self.modality, self.python_executable = modality.upper(), python_executable
        if self.modality not in ("CT", "MR"):
            raise ValueError("modality must be CT or MR")

    def to_anatomy_collection(self, *, names: Iterable[str] | None = None) -> AnatomyCollection:
        """Run inference for ``names`` (default: every supported catalog label) and mesh the output.

        Raises ``ImportError`` if the bundle is missing and ``ValueError`` for unsupported names.
        """
        config, meta = self.root / "configs/inference.json", self.root / "configs/metadata.json"
        if not config.is_file():
            raise ImportError("NV-Segment-CTMR bundle is missing: clone https://github.com/NVIDIA-Medtech/"
                              "NV-Segment-CTMR and set bundle_root to its inner NV-Segment-CTMR directory.")
        dataset = "CT" if self.modality == "CT" else "MRI"
        definitions = json.loads((self.root / "configs/label_dict.json").read_text())
        supported = _supported({v["index"]: k for k, v in definitions.items() if dataset in v.get("datasets", [])}, names)
        if not supported:
            raise ValueError(f"No catalog labels supported by NV-Segment for {self.modality}")
        with TemporaryDirectory(prefix="patient-nvsegment-") as temp_dir:
            temp = Path(temp_dir)
            nib.save(self.image, temp / "image.nii.gz")
            overrides = json.loads(config.read_text())
            overrides.update(
                bundle_root=str(self.root), input_dict={"image": str(temp / "image.nii.gz")},
                modality=f"{dataset}_BODY", everything_labels=list(supported),
                output_dir=str(temp / "masks"), output_postfix="segmentation", separate_folder=False,
            )
            (temp / "inference.json").write_text(json.dumps(overrides))
            if self.python_executable is None:
                with _backend(self.root):
                    from monai.bundle import run

                    run(config_file=str(temp / "inference.json"), meta_file=str(meta))
            else:
                subprocess.run([str(self.python_executable), "-m", "monai.bundle", "run", "--config_file",
                                str(temp / "inference.json"), "--meta_file", str(meta)], cwd=self.root, check=True)
            outputs = list((temp / "masks").rglob("*.nii.gz"))
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one NV-Segment output, found {len(outputs)}")
            result = _load_nifti(outputs[0])
            if result.shape != self.image.shape or not np.allclose(
                nifti_affine_m(result), nifti_affine_m(self.image), atol=1e-6
            ):
                raise ValueError("NV-Segment output must match the input image physical grid")
            return segmentation_anatomy(result, supported, names=names)


def _generate(seed: "int | str", mask_output: "str | Path", ct_output: "str | Path") -> None:
    """Run upstream NV-Generate rflow-ct inference from its checkout; self-contained for -c.

    Data files resolve exactly as upstream does: ``datasets/...`` entries live under
    ``$MONAI_DATA_DIRECTORY`` (or a fresh directory when it is unset), downloaded by
    upstream's ``download_model_data``. Inference then runs against that same root.
    """
    import json
    import os
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    from scripts import inference, sample
    from scripts.download_model_data import download_model_data

    folder = Path(mask_output).parent
    data_root = os.environ.get("MONAI_DATA_DIRECTORY") or tempfile.mkdtemp(prefix="data-", dir=folder)
    os.makedirs(data_root, exist_ok=True)
    download_model_data("rflow-ct", data_root)  # Skips files that already exist.
    environment = json.loads(Path("configs/environment_rflow-ct.json").read_text())
    # Upstream's rule; absolute paths survive its own os.path.join(root_dir, value) unchanged.
    environment = {k: os.path.join(data_root, v) if "datasets/" in v else v for k, v in environment.items()}
    config = json.loads(Path("configs/config_infer.json").read_text())
    # A size condition triggers fresh mask diffusion instead of selecting a cached mask.
    conditions = json.loads(Path(environment["all_anatomy_size_conditions_json"]).read_text())
    config.update(num_output_samples=1, image_output_ext=".nii.gz", label_output_ext=".nii.gz",
                  controllable_anatomy_size=[["liver", conditions[0]["organ_size"][1]]])
    environment["output_dir"] = str(folder / "generated")
    (folder / "environment.json").write_text(json.dumps(environment))
    (folder / "inference.json").write_text(json.dumps(config))
    # Upstream keeps only the conditioning organ in its saved mask; keep every label.
    original_filter, original_argv = sample.filter_mask_with_organs, sys.argv
    original_root = os.environ.get("MONAI_DATA_DIRECTORY")
    sample.filter_mask_with_organs = lambda labels, organs: labels
    sys.argv = ["inference", "-t", "configs/config_network_rflow.json", "-e", str(folder / "environment.json"),
                "-i", str(folder / "inference.json"), "--random-seed", str(seed), "--version", "rflow-ct"]
    os.environ["MONAI_DATA_DIRECTORY"] = data_root  # Inference resolves (and downloads into) the same root.
    try:
        inference.main()
    finally:
        sample.filter_mask_with_organs, sys.argv = original_filter, original_argv
        if original_root is None:
            os.environ.pop("MONAI_DATA_DIRECTORY", None)
        else:
            os.environ["MONAI_DATA_DIRECTORY"] = original_root
    images = list((folder / "generated").glob("*_image.nii.gz"))
    if len(images) != 1:
        raise RuntimeError(f"Expected one generated CT, found {len(images)}")
    mask = images[0].with_name(images[0].name.replace("_image.nii.gz", "_label.nii.gz"))
    if not mask.is_file():
        raise RuntimeError("Generated CT is missing its paired segmentation")
    shutil.copyfile(mask, mask_output)
    shutil.copyfile(images[0], ct_output)


class NVGenerateImporter:
    """Generate a synthetic patient (paired CT + labels) with NV-Generate-CTMR, then mesh it.

    Install ``patient-digital-twin[nvgenerate]`` and clone the upstream checkout with weights.
    After import, ``ct_scan`` holds the generated CT and ``seed`` the random seed used.

    Example:
        ``generator = NVGenerateImporter(source_root="NV-Generate-CTMR")``
        ``body = HumanBody(generator.to_anatomy_collection(names=["aorta"]))``
        ``body.attach_scan(generator.ct_scan)``
    """

    def __init__(self, *, source_root: str | Path | None = None, python_executable: str | Path | None = None) -> None:
        """``source_root`` defaults to ``$NV_GENERATE_ROOT``; ``python_executable`` runs out of process."""
        self.root = Path(source_root or os.environ.get("NV_GENERATE_ROOT", ".")).resolve()
        self.python_executable = python_executable
        self.seed: int | None = None
        self.ct_scan: ScanVolume | None = None

    def to_anatomy_collection(self, *, names: Iterable[str] | None = None) -> AnatomyCollection:
        """Generate a new patient with a fresh random seed and mesh ``names`` (default: all)."""
        from .scan_volume import from_nifti

        self.ct_scan = None
        if not (self.root / "scripts/inference.py").is_file():
            raise ImportError("NV-Generate-CTMR source is missing: clone "
                              "https://github.com/NVIDIA-Medtech/NV-Generate-CTMR and pass source_root.")
        labels = json.loads((self.root / "configs/label_dict.json").read_text())
        labelmap = {value: name for name, value in labels.items()}
        _supported(labelmap, names)
        self.seed = seed = secrets.randbits(32)
        with TemporaryDirectory(prefix="patient-nvgenerate-") as temp:
            mask_path, ct_path = Path(temp) / "mask.nii.gz", Path(temp) / "ct.nii.gz"
            if self.python_executable is None:
                with _backend(self.root):
                    _generate(seed, mask_path, ct_path)
            else:
                code = inspect.getsource(_generate) + "\nimport sys\n_generate(*sys.argv[1:])\n"
                subprocess.run([str(self.python_executable), "-c", code, str(seed), str(mask_path),
                                str(ct_path)], cwd=self.root, check=True)
            mask_image, ct_image = _load_nifti(mask_path), _load_nifti(ct_path)
            if mask_image.shape != ct_image.shape or not np.allclose(
                nifti_affine_m(mask_image), nifti_affine_m(ct_image)
            ):
                raise ValueError("Generated CT and segmentation must share their physical grid")
            scan = from_nifti(ct_path)
            scan.metadata["source"] = {"kind": "generated", "backend": "NV-Generate-CTMR", "seed": self.seed}
            body = segmentation_anatomy(mask_image, labelmap, names=names)
        self.ct_scan = scan
        return body


class SimpleImporter:
    """Load named STL/OBJ meshes (XYZ meters; requires ``trimesh``) as anatomy, without resizing.

    Example:
        ``SimpleImporter({"liver": "liver.stl"}, mesh_to_body={"liver": placement}).to_anatomy_collection()``
    """

    def __init__(
        self,
        meshes: Mapping[str, str | Path],
        *,
        mesh_to_body: Mapping[str, ArrayLike] | None = None,
        body_to_imaging: ArrayLike | None = None,
    ) -> None:
        """Args: ``meshes`` name -> file; ``mesh_to_body`` optional rigid 4x4 per name (default
        identity); ``body_to_imaging`` optional registration stored on the collection."""
        if not meshes:
            raise ValueError("Provide at least one anatomy name and mesh path")
        self.meshes, self.transforms = dict(meshes), dict(mesh_to_body or {})
        self.body_to_imaging = body_to_imaging

    def to_anatomy_collection(self) -> AnatomyCollection:
        """Load every mesh; names are canonicalized and must be catalog names."""
        import trimesh

        names = {raw: canonical_name(raw) for raw in self.meshes}
        transforms = {canonical_name(n): value for n, value in self.transforms.items()}
        if len(set(names.values())) != len(names) or set(transforms) - set(names.values()):
            raise ValueError("Mesh names must be unique and every transform must name a mesh")
        body = AnatomyCollection.from_names(names.values())
        body.body_to_imaging = self.body_to_imaging
        for raw, name in names.items():
            if Path(self.meshes[raw]).suffix.lower() not in (".stl", ".obj"):
                raise ValueError("Supported mesh formats: .stl, .obj")
            mesh: Any = cast(Any, trimesh.load(str(self.meshes[raw]), force="scene", process=False)).to_geometry()
            structure = body.structures[name]
            structure.vertices, structure.faces = validate_triangles(mesh.vertices, mesh.faces.astype(np.int64), name=name)
            structure.local_to_body = rigid_transform(transforms.get(name, np.eye(4)))
            structure.local_to_world = structure.local_to_body.copy()
        return body
