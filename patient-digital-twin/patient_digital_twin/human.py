# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small application-facing API for a patient's anatomy and imaging."""

from __future__ import annotations

from .anatomy import AnatomyCollection
from .geometry import transform_points
from .imaging import ImagingVolume


class HumanBody:
    """A patient has anatomy and optional attached imaging.

    Use body.anatomy to configure mesh availability and access system views.
    """

    def __init__(
        self,
        anatomy=None,
        *,
        configuration=None,
    ):
        """Wrap imported anatomy (or a structure dictionary)."""
        self.anatomy = (
            anatomy
            if isinstance(anatomy, AnatomyCollection)
            else AnatomyCollection(anatomy)
        )
        self.imaging: ImagingVolume | None = None
        if configuration is not None:
            self.anatomy.configure(configuration)

    def attach_imaging(
        self,
        volume,
        *,
        voxel_to_imaging,
        body_to_imaging=None,
        source_path=None,
        modality="CT",
        source_scan=None,
    ):
        """Attach a NumPy ZYX volume without loading a file or moving anatomy.

        voxel_to_imaging maps XYZ voxel indices to RAS meters. body_to_imaging
        maps the anatomy body frame to that same frame; if omitted, use the
        imported anatomy's source registration. Supply it explicitly for an
        image in a different frame. CT values must already be HU. source_path
        is optional provenance only. Failed validation preserves prior imaging.
        """
        registration = (
            self.anatomy.body_to_imaging if body_to_imaging is None else body_to_imaging
        )
        if registration is None:
            raise ValueError("attach_imaging requires body_to_imaging registration")
        self.imaging = ImagingVolume(
            volume, voxel_to_imaging, registration, source_path, modality, source_scan
        )
        return self.imaging

    def attach_scan(self, scan, *, body_to_imaging=None, source_path=None):
        """Attach a scan artifact while retaining its native array order and units."""
        return self.attach_imaging(
            scan.values_kji,
            voxel_to_imaging=scan.ijk_to_ras_m,
            body_to_imaging=body_to_imaging,
            source_path=source_path,
            source_scan=scan,
        )

    def imaging_vertices(self, name):
        """Recover an enabled structure's original scan placement, before posing."""
        if self.imaging is None:
            raise ValueError("Call attach_imaging before requesting imaging vertices")
        points = self.anatomy.structures[name].body_vertices
        return (
            None
            if points is None
            else transform_points(points, self.imaging.body_to_imaging)
        )

    def export_to_usd(self, path):
        """Export retained anatomy, centerlines, and optional CT to USD.

        Requires usd-core. Hidden meshes retain their geometry and visibility.
        Stored centerlines and attached CT voxels become custom attributes.
        Structure transforms are preserved in a meter-scale, Z-up stage.
        Returns the written Path.
        """
        from .exporters.usd import export_to_usd

        return export_to_usd(self, path)

    def export_patient_twin(self, output, **options):
        """Export the original scan-frame bundle consumed by i4h-workflows.

        Includes retained meshes, centerline assets, and optional imaging/skin.
        Without a source registration, uses the body frame. Without CT, exports
        anatomy alone. Output must be a new directory.
        """
        from .exporters.patient_twin import export_patient_twin

        return export_patient_twin(self, output, **options)

    def extract_topology(self, *, names=None, spacing_m=0.0015):
        """Store local centerlines for all retained vessel and airway meshes.

        Includes disabled anatomy and the catalog's composite portal-vein mask.
        Each mesh is voxelized on a bounded spacing_m grid in its local frame
        and skeletonized. Missing meshes are skipped. Results are committed only
        after every extraction succeeds; failures identify the offending
        anatomy. Optional names limits extraction to selected items.
        """
        import numpy as np

        from .catalog import is_vessel
        from .structures import Kind
        from .topology import extract_centerlines

        if not np.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("spacing_m must be positive and finite")
        selected = set(self.anatomy.structures) if names is None else set(names)
        unknown = selected - self.anatomy.structures.keys()
        if unknown:
            raise KeyError(f"Unknown anatomy: {sorted(unknown)}")
        results = {}
        for name, structure in self.anatomy.structures.items():
            if name not in selected:
                continue
            if not (is_vessel(name, structure.kind) or structure.kind == Kind.AIRWAY):
                continue
            vertices, faces = structure.mesh.vertices, structure.mesh.faces
            if vertices is None or faces is None:
                continue
            origin = vertices.min(0) - 2 * spacing_m
            shape = np.ceil((vertices.max(0) - origin) / spacing_m).astype(int) + 3
            try:
                results[name] = extract_centerlines(
                    vertices,
                    faces,
                    shape_zyx=tuple(shape[::-1]),
                    spacing_zyx_m=(spacing_m,) * 3,
                    origin_xyz_m=origin,
                )
            except ImportError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Centerline extraction failed for {name}: {exc}"
                ) from exc
        for name, graph in results.items():
            self.anatomy.structures[name].centerline = graph
        return results
