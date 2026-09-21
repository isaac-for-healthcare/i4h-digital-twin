# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small application-facing API for a patient's anatomy and external body."""

from __future__ import annotations

from .anatomy import AnatomicalSystem, AnatomyCollection
from .configuration import AnatomyConfiguration
from .geometry import rigid_transform, transform_points
from .soma_body import SomaRepresentation
from .structures import AnatomicalStructure, Kind, System


class HumanBody:
    """A patient has anatomy and an optional SOMA representation.

    Use configure_anatomy() or the enabled setters to make meshes empty without
    losing their geometry. Use body.anatomy for system views and body.soma for
    model-specific fitting, registration, parameters, and diagnostic reports.
    """

    def __init__(
        self,
        structures=None,
        source_path=None,
        landmarks=None,
        *,
        configuration=None,
        body_to_imaging=None,
    ):
        """Create an empty patient or wrap a dictionary of named structures.

        Pass body-frame XYZ-meter landmarks when automatic alignment is
        needed. Configuration is applied after structures exist. Attach a model
        with attach_soma(), not with constructor model/registration arguments.
        """
        self.anatomy = AnatomyCollection(structures)
        self.soma = SomaRepresentation(self.anatomy)
        self.source_path = source_path
        self.body_to_imaging = body_to_imaging
        if landmarks is not None:
            self.landmarks = landmarks
        if configuration is not None:
            self.configure_anatomy(configuration)

    @property
    def body_to_imaging(self):
        """Optional rigid transform from the imported body frame to source imaging."""
        return None if self._body_to_imaging is None else self._body_to_imaging.copy()

    @body_to_imaging.setter
    def body_to_imaging(self, value):
        """Set or discard the source-frame relationship without moving anatomy."""
        self._body_to_imaging = None if value is None else rigid_transform(value).copy()

    def imaging_vertices(self, name):
        """Recover an enabled structure's original scan placement, before posing."""
        if self._body_to_imaging is None:
            raise ValueError("No body_to_imaging transform is available")
        points = self.structures[name].body_vertices
        return (
            None if points is None else transform_points(points, self._body_to_imaging)
        )

    @property
    def structures(self) -> dict[str, AnatomicalStructure]:
        """Named structures, including disabled and unsegmented anatomy."""
        return self.anatomy.structures

    def export_to_usd(self, path, *, skin_opacity=0.15):
        """Export the current SOMA skin and individually selectable anatomy meshes.

        Requires optional usd-core. Disabled meshes are retained as invisible.
        Returns the written Path; hide /HumanBody/Exterior to inspect organs.
        """
        from .usd import export_to_usd

        return export_to_usd(self, path, skin_opacity=skin_opacity)

    def export_patient_twin(self, output, *, ct_path, **options):
        """Export the original scan-frame bundle consumed by i4h-workflows.

        Includes attenuation, CT exterior, internal meshes and vessel topology.
        Requires a CT and body_to_imaging; output must be a new directory.
        """
        from .patient_twin import export_patient_twin

        return export_patient_twin(self, output, ct_path=ct_path, **options)

    def extract_topology(self, *, names=None, **options):
        """Store local centerlines for all retained vessel and airway meshes.

        Includes disabled anatomy and the catalog's composite portal-vein mask.
        Options are passed to topology.extract_centerlines (grid coordinates must
        be in each mesh's local frame). Missing meshes are skipped. Results are
        committed only after every extraction succeeds; failures identify the
        offending anatomy. Optional names limits extraction to selected items.
        """
        from .topology import extract_centerlines

        selected = set(self.structures) if names is None else set(names)
        unknown = selected - self.structures.keys()
        if unknown:
            raise KeyError(f"Unknown anatomy: {sorted(unknown)}")
        results = {}
        for name, structure in self.structures.items():
            if name not in selected:
                continue
            if structure.kind not in {Kind.VESSEL, Kind.AIRWAY} and name not in {
                "trachea",
                "portal_vein_and_splenic_vein",
            }:
                continue
            if structure.mesh.vertices is None or structure.mesh.faces is None:
                continue
            try:
                results[name] = extract_centerlines(
                    structure.mesh.vertices, structure.mesh.faces, **options
                )
            except ImportError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Centerline extraction failed for {name}: {exc}"
                ) from exc
        for name, graph in results.items():
            self.structures[name].centerline = graph
        return results

    @property
    def anatomy_configuration(self) -> AnatomyConfiguration:
        """Read the immutable mesh policy; use configure_anatomy() to replace it."""
        return self.anatomy.configuration

    def configure_anatomy(self, source) -> None:
        """Replace mesh settings from a YAML path, mapping, or AnatomyConfiguration."""
        self.anatomy.configure(source)

    def set_anatomy_enabled(self, enabled: bool) -> None:
        """Enable/empty all imported anatomy; the external SOMA skin is unaffected."""
        self.anatomy.set_enabled(enabled)

    def set_system_enabled(self, system: System | str, enabled: bool) -> None:
        """Toggle a system by enum or name; explicit structure overrides survive."""
        self.anatomy.set_system_enabled(system, enabled)

    def set_structure_enabled(self, name: str, enabled: bool) -> None:
        """Override one existing structure; the master disable still takes priority."""
        self.anatomy.set_structure_enabled(name, enabled)

    def system(self, name: System | str) -> AnatomicalSystem:
        """Return a live system view, e.g. body.system("digestive").is_empty."""
        return self.anatomy.system(name)

    def select(
        self,
        *,
        kind: Kind | None = None,
        system: System | str | None = None,
        region: str | None = None,
        include_empty: bool = True,
    ) -> list[AnatomicalStructure]:
        """Find anatomy by kind and catalog membership, optionally excluding empty structures."""
        return self.anatomy.select(
            kind=kind,
            system=system,
            region=region,
            include_empty=include_empty,
        )

    def attach_soma(self, layer=None, **options):
        """Attach and align an external body; see SomaRepresentation.attach_soma."""
        return self.soma.attach_soma(layer, **options)

    def pose(self, poses=None, *, transl=None, **parameters):
        """Move the external body and its attached anatomy together."""
        return self.soma.pose(poses, transl=transl, **parameters)

    def check_containment(self, **options) -> dict[str, dict]:
        """Check enabled anatomy against the skin; empty structures are excluded."""
        return self.soma.check_containment(**options)

    # Compatibility conveniences. Advanced implementation and state live in soma.
    def update_from_soma(self, output) -> None:
        """Synchronize anatomy after SOMA two-phase evaluation (layer.pose())."""
        self.soma.update_from_soma(output)

    def fit_soma_shape(self, poses=(), **options) -> dict:
        """Fit external shape around rigid anatomy; supply required poses and recheck containment."""
        return self.soma.fit_soma_shape(poses, **options)

    def fit_bone_anchors(self, poses=(), **options) -> dict:
        """Fit fixed rigid bone offsets after shape fitting, without changing bone geometry."""
        return self.soma.fit_bone_anchors(poses, **options)

    @property
    def soma_layer(self):
        """Attached SOMALayer, or None; direct forward calls also update anatomy."""
        return self.soma.soma_layer

    @property
    def soma_body(self):
        """Latest skin and joint-frame snapshot, or None before attachment."""
        return self.soma.soma_body

    @property
    def soma_parameters(self) -> dict:
        """Copy fitted scan parameters for editing; pass changed poses to pose()."""
        return self.soma.soma_parameters

    @property
    def soma_configuration(self) -> dict:
        """JSON-ready attachment arguments; reload with attach_soma(**config)."""
        return self.soma.soma_configuration

    @property
    def body_to_soma(self):
        """Read the rigid body-frame to SOMA-world registration in meters."""
        return self.soma.body_to_soma

    @property
    def landmarks(self):
        """Body-frame XYZ-meter landmarks used for automatic registration."""
        return self.soma.landmarks

    @landmarks.setter
    def landmarks(self, value):
        """Set measured landmarks before attach_soma(); this does not refit a model."""
        self.soma.landmarks = value

    @property
    def alignment_errors_m(self) -> dict:
        """Per-landmark registration residuals in meters; inspect after attachment."""
        return self.soma.alignment_errors_m

    @property
    def shape_fit_report(self) -> dict:
        """Identity-fitting history; containment must still be checked separately."""
        return self.soma.shape_fit_report

    @property
    def bone_fit_report(self) -> dict:
        """Fitted rigid bone offsets and residuals; no bone scaling is performed."""
        return self.soma.bone_fit_report
