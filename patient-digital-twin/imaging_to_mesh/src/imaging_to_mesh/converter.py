# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public medical-image to mesh conversion APIs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

from .labels import CATEGORY_LABELS
from .mesh import mask_to_mesh, write_obj
from .models import ConversionResult, MeshArtifact
from .usd import write_usd


def convert_nrrd_to_nifti(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert an NRRD image to NIfTI.

    Install the optional dependency first with ``pip install imaging-to-mesh[nrrd]``.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("NRRD support requires `pip install imaging-to-mesh[nrrd]`.") from exc

    source = Path(input_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.ReadImage(str(source)), str(output))
    return output


def load_labelmap(
    input_path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Load a NIfTI or NRRD labelmap as ``(array_zyx, spacing_zyx_mm, origin_xyz_mm)``."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    if source.name.endswith((".nii", ".nii.gz")):
        image = nib.load(str(source))
        labels_xyz = np.asarray(image.dataobj)
        spacing_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
        origin_xyz = tuple(float(v) for v in image.affine[:3, 3])
        return labels_xyz.transpose(2, 1, 0), spacing_xyz[::-1], origin_xyz

    if source.suffix.lower() == ".nrrd":
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("NRRD support requires `pip install imaging-to-mesh[nrrd]`.") from exc
        image = sitk.ReadImage(str(source))
        return (
            sitk.GetArrayFromImage(image),
            tuple(float(v) for v in image.GetSpacing()[::-1]),
            tuple(float(v) for v in image.GetOrigin()),
        )

    raise ValueError(f"Unsupported input format: {source}. Expected .nii, .nii.gz, or .nrrd.")


def convert_mask_to_usd(
    mask_zyx: np.ndarray,
    output_path: str | Path,
    *,
    name: str = "Anatomy",
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    write_obj_file: bool = True,
) -> ConversionResult:
    """Convert a binary NumPy mask directly to OBJ and OpenUSD."""
    output = Path(output_path)
    vertices, faces = mask_to_mesh(
        mask_zyx,
        spacing_zyx_mm=spacing_zyx_mm,
        origin_xyz_mm=origin_xyz_mm,
    )
    obj_path = output.with_name(f"{output.stem}_{name}.obj")
    if write_obj_file:
        write_obj(vertices, faces, obj_path)
    write_usd({name: (vertices, faces)}, output)
    artifact = MeshArtifact(
        name=name,
        obj_path=obj_path,
        vertex_count=len(vertices),
        face_count=len(faces),
    )
    return ConversionResult(usd_path=output, meshes=(artifact,))


def convert_segmentation_array(
    labels_zyx: np.ndarray,
    output_dir: str | Path,
    *,
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    categories: Mapping[str, Sequence[int]] = CATEGORY_LABELS,
    usd_filename: str = "all_organs.usd",
) -> ConversionResult:
    """Convert a labeled ZYX segmentation array into per-category OBJ and one USD."""
    labels = np.asarray(labels_zyx)
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D labelmap, got shape {labels.shape}")

    output = Path(output_dir)
    obj_dir = output / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    artifacts = []

    for name, label_values in categories.items():
        mask = np.isin(labels, tuple(label_values))
        if not np.any(mask):
            continue
        vertices, faces = mask_to_mesh(
            mask,
            spacing_zyx_mm=spacing_zyx_mm,
            origin_xyz_mm=origin_xyz_mm,
        )
        obj_path = write_obj(vertices, faces, obj_dir / f"{name}.obj")
        meshes[name] = (vertices, faces)
        artifacts.append(MeshArtifact(name, obj_path, len(vertices), len(faces)))

    if not meshes:
        raise ValueError("The segmentation contains none of the requested anatomical labels.")

    usd_path = write_usd(meshes, output / usd_filename)
    return ConversionResult(usd_path=usd_path, meshes=tuple(artifacts))


def convert_segmentation_file(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    categories: Mapping[str, Sequence[int]] = CATEGORY_LABELS,
) -> ConversionResult:
    """Convert one NIfTI/NRRD segmentation labelmap to OBJ and OpenUSD."""
    source = Path(input_path)
    labels, spacing, origin = load_labelmap(source)
    stem = source.name.removesuffix(".nii.gz").removesuffix(".nii").removesuffix(".nrrd")
    destination = Path(output_dir) if output_dir is not None else source.parent / stem
    result = convert_segmentation_array(
        labels,
        destination,
        spacing_zyx_mm=spacing,
        origin_xyz_mm=origin,
        categories=categories,
    )
    return ConversionResult(result.usd_path, result.meshes, source)


def convert_path(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    pattern: str = r".*label\.(nii(\.gz)?|nrrd)$",
) -> tuple[ConversionResult, ...]:
    """Convert one labelmap or matching labelmaps directly inside a directory."""
    source = Path(input_path)
    if source.is_file():
        return (convert_segmentation_file(source, output_dir),)
    if not source.is_dir():
        raise FileNotFoundError(source)

    regex = re.compile(pattern)
    matches = sorted(path for path in source.iterdir() if path.is_file() and regex.fullmatch(path.name))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {source}")

    root = Path(output_dir) if output_dir is not None else source
    return tuple(convert_segmentation_file(path, root / path.name.split(".")[0]) for path in matches)
