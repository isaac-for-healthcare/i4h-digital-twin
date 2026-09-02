# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reorientation of CT volumes into the canonical anatomical frame.

Scanners write slices in whatever order the acquisition produced, so the anatomical
meaning of a raw array axis varies between studies: two CTs of the same anatomy can
differ by axis permutations and flips while both are perfectly valid DICOM. Downstream
consumers (C-arm pose conventions, centerline coordinates, contact geometry) assume a
fixed patient frame, so orientation has to be resolved once, at ingest.

Anatomical frame used by this package (``CANONICAL_FRAME = "LPS"``):

* array axis 0 (slice, ``z``) increases toward the patient's **Superior** (head),
* array axis 1 (row, ``y``) increases toward the patient's **Posterior** (back),
* array axis 2 (column, ``x``) increases toward the patient's **Left**.

Mapping array axes to world axes as ``(x, y, z) = (axis 2, axis 1, axis 0)`` makes the
world axes Left / Posterior / Superior, which is the DICOM patient coordinate system.

``direction`` matrices here follow the ITK convention returned by
``SimpleITK.Image.GetDirection()``: a row-major 3x3 whose **columns** are the unit
patient-space directions of the index axes ``(i, j, k)``, i.e. of array axes
``(2, 1, 0)``, expressed in LPS. NIfTI affines are in RAS and must be converted with
:func:`affine_to_lps` before use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANONICAL_FRAME = "LPS"

# Canonical anatomical direction of each array axis, as (patient axis index, sign).
# Patient axis indices are LPS: 0 = Left(+)/Right(-), 1 = Posterior(+)/Anterior(-),
# 2 = Superior(+)/Inferior(-).
_CANONICAL_TARGETS: tuple[tuple[int, int], ...] = ((2, 1), (1, 1), (0, 1))

# Letter for the anatomical direction an axis points toward, indexed by [patient axis][sign].
_AXIS_LETTERS: tuple[tuple[str, str], ...] = (("R", "L"), ("A", "P"), ("I", "S"))

# Below this dot product with the nearest patient axis, an acquisition is oblique enough
# that permutations and flips alone cannot align it and resampling would be required.
_OBLIQUITY_WARN_COSINE = 0.95


@dataclass(frozen=True)
class CanonicalVolume:
    """A CT volume reoriented into the canonical LPS frame.

    Attributes:
        hu_zyx: Reoriented HU volume, C-contiguous.
        spacing_zyx_mm: Voxel spacing permuted to match ``hu_zyx``.
        origin_xyz_mm: Patient-space (LPS) position of voxel ``[0, 0, 0]`` after
            reorientation, or None if the input origin was unknown.
        direction: Row-major 3x3 direction cosines of the reoriented volume, columns in
            index order ``(i, j, k)``. Exactly the identity for axis-aligned inputs; the
            residual rotation for oblique ones.
        source_code: Anatomical directions the source array axes ``(0, 1, 2)`` increased
            toward, e.g. ``"IPL"`` for a supine axial series whose first slice is at the
            head. ``"SPL"`` means the source was already canonical.
        permutation: Source array axis feeding each canonical axis ``(0, 1, 2)``.
        flipped_axes: Canonical axes whose source axis pointed the opposite way.
        max_obliquity_deg: Largest angle between a reoriented axis and its canonical
            patient axis. Zero for axis-aligned acquisitions.
    """

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float] | None
    direction: tuple[float, ...]
    source_code: str
    permutation: tuple[int, int, int]
    flipped_axes: tuple[int, ...]
    max_obliquity_deg: float

    @property
    def is_identity(self) -> bool:
        """Return whether the source volume was already in the canonical frame."""
        return self.permutation == (0, 1, 2) and not self.flipped_axes

    @property
    def is_oblique(self) -> bool:
        """Return whether the acquisition is too oblique to align by permute and flip."""
        return self.max_obliquity_deg > np.degrees(np.arccos(_OBLIQUITY_WARN_COSINE))

    def summary(self) -> str:
        """Return a one-line human-readable description of what was applied."""
        if self.is_identity:
            return f"already canonical ({CANONICAL_FRAME}: {self.source_code})"
        return (
            f"{self.source_code} -> {CANONICAL_FRAME} canonical "
            f"(permutation={self.permutation}, flipped_axes={self.flipped_axes or '()'}, "
            f"obliquity={self.max_obliquity_deg:.1f} deg)"
        )


def _direction_matrix(direction: tuple[float, ...] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError(f"direction contains non-finite values: {matrix.tolist()}")
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms < 1e-9):
        raise ValueError(f"direction has a degenerate column with zero length: {matrix.tolist()}")
    return matrix / norms


def _axis_vector(matrix: np.ndarray, array_axis: int) -> np.ndarray:
    """Return the patient-space direction of ``array_axis``, columns being index axes."""
    return matrix[:, 2 - array_axis]


def _nearest_patient_axes(matrix: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Assign each array axis a distinct patient axis and sign.

    Claims the largest direction cosines first so that a near-axis-aligned axis is never
    displaced by an oblique one competing for the same patient axis.

    Args:
        matrix: Column-normalised 3x3 direction cosines.

    Returns:
        ``(patient_axis, sign)`` per array axis ``(0, 1, 2)``.
    """
    cosines = np.array([_axis_vector(matrix, axis) for axis in range(3)])  # [array axis, patient axis]
    assignment: dict[int, tuple[int, int]] = {}
    remaining = np.abs(cosines).copy()

    for _ in range(3):
        array_axis, patient_axis = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        sign = 1 if cosines[array_axis, patient_axis] >= 0.0 else -1
        assignment[int(array_axis)] = (int(patient_axis), sign)
        remaining[array_axis, :] = -1.0
        remaining[:, patient_axis] = -1.0

    return tuple(assignment[axis] for axis in range(3))


def orientation_code(direction: tuple[float, ...] | np.ndarray) -> str:
    """Describe which anatomical direction each array axis increases toward.

    Args:
        direction: Row-major 3x3 direction cosines in LPS, columns in index order
            ``(i, j, k)``.

    Returns:
        Three letters from ``RLAPIS`` for array axes ``(0, 1, 2)``. The canonical frame
        is ``"SPL"``.
    """
    axes = _nearest_patient_axes(_direction_matrix(direction))
    return "".join(_AXIS_LETTERS[patient_axis][1 if sign > 0 else 0] for patient_axis, sign in axes)


def to_canonical_lps(
    hu_zyx: np.ndarray,
    direction: tuple[float, ...] | np.ndarray | None,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float] | None = None,
) -> CanonicalVolume:
    """Permute and flip a CT volume into the canonical LPS frame.

    Nearest-axis reorientation only: the volume is never resampled, so an oblique
    acquisition keeps a residual rotation reported as ``max_obliquity_deg``.

    Args:
        hu_zyx: HU volume in ``(axis 0, axis 1, axis 2)`` array order.
        direction: Row-major 3x3 direction cosines in LPS, columns in index order
            ``(i, j, k)``. When None the volume is assumed to be canonical already and
            is returned unchanged.
        spacing_zyx_mm: Voxel spacing matching the axes of ``hu_zyx``.
        origin_xyz_mm: Patient-space (LPS) position of voxel ``[0, 0, 0]``, if known.

    Returns:
        The reoriented volume together with the transform that was applied.

    Raises:
        ValueError: If ``hu_zyx`` is not 3D, or the direction matrix is degenerate.
    """
    if hu_zyx.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {hu_zyx.shape}")

    if direction is None:
        return CanonicalVolume(
            hu_zyx=np.ascontiguousarray(hu_zyx),
            spacing_zyx_mm=tuple(float(s) for s in spacing_zyx_mm),
            origin_xyz_mm=tuple(float(v) for v in origin_xyz_mm) if origin_xyz_mm is not None else None,
            direction=tuple(np.eye(3).flatten()),
            source_code=_canonical_code(),
            permutation=(0, 1, 2),
            flipped_axes=(),
            max_obliquity_deg=0.0,
        )

    matrix = _direction_matrix(direction)
    axes = _nearest_patient_axes(matrix)
    source_code = "".join(_AXIS_LETTERS[patient_axis][1 if sign > 0 else 0] for patient_axis, sign in axes)

    by_patient_axis = {patient_axis: (array_axis, sign) for array_axis, (patient_axis, sign) in enumerate(axes)}
    permutation = tuple(by_patient_axis[patient_axis][0] for patient_axis, _ in _CANONICAL_TARGETS)
    signs = tuple(
        by_patient_axis[patient_axis][1] * target_sign for patient_axis, target_sign in _CANONICAL_TARGETS
    )
    flipped_axes = tuple(axis for axis, sign in enumerate(signs) if sign < 0)

    reoriented = np.transpose(hu_zyx, permutation)
    if flipped_axes:
        reoriented = np.flip(reoriented, axis=flipped_axes)
    reoriented = np.ascontiguousarray(reoriented)

    spacing = tuple(float(spacing_zyx_mm[source_axis]) for source_axis in permutation)

    # Columns of the reoriented direction matrix, in index order (i, j, k) = axes (2, 1, 0).
    columns = [signs[axis] * _axis_vector(matrix, permutation[axis]) for axis in range(3)]
    canonical_matrix = np.stack(columns[::-1], axis=1)

    origin = _reoriented_origin(
        matrix=matrix,
        shape_zyx=hu_zyx.shape,
        spacing_zyx_mm=spacing_zyx_mm,
        origin_xyz_mm=origin_xyz_mm,
        permutation=permutation,
        signs=signs,
    )

    cosines = [float(np.dot(columns[axis], _unit_patient_axis(_CANONICAL_TARGETS[axis]))) for axis in range(3)]
    max_obliquity_deg = float(np.degrees(np.arccos(np.clip(min(cosines), -1.0, 1.0))))

    return CanonicalVolume(
        hu_zyx=reoriented,
        spacing_zyx_mm=spacing,
        origin_xyz_mm=origin,
        direction=tuple(float(v) for v in canonical_matrix.flatten()),
        source_code=source_code,
        permutation=permutation,
        flipped_axes=flipped_axes,
        max_obliquity_deg=max_obliquity_deg,
    )


def affine_to_lps(
    affine: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float]]:
    """Convert a NIfTI RAS affine into LPS direction cosines, spacing and origin.

    Args:
        affine: 4x4 voxel-index-to-RAS affine, as returned by ``nibabel``.

    Returns:
        Tuple of (row-major 3x3 direction cosines in LPS with columns in index order,
        spacing in index order ``(i, j, k)`` in mm, LPS origin of voxel ``[0, 0, 0]``).

    Raises:
        ValueError: If the affine is not 4x4 or has a degenerate rotation block.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine, got shape {affine.shape}")

    rotation = affine[:3, :3]
    spacing = np.linalg.norm(rotation, axis=0)
    if np.any(spacing < 1e-9):
        raise ValueError(f"Affine has a degenerate rotation block: {rotation.tolist()}")

    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    direction = ras_to_lps @ (rotation / spacing)
    origin = ras_to_lps @ affine[:3, 3]

    return (
        tuple(float(v) for v in direction.flatten()),
        tuple(float(v) for v in spacing),
        tuple(float(v) for v in origin),
    )


def _canonical_code() -> str:
    return "".join(_AXIS_LETTERS[patient_axis][1 if sign > 0 else 0] for patient_axis, sign in _CANONICAL_TARGETS)


def _unit_patient_axis(target: tuple[int, int]) -> np.ndarray:
    patient_axis, sign = target
    vector = np.zeros(3)
    vector[patient_axis] = float(sign)
    return vector


def _reoriented_origin(
    matrix: np.ndarray,
    shape_zyx: tuple[int, ...],
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float] | None,
    permutation: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> tuple[float, float, float] | None:
    """Return the patient-space position of the reoriented volume's first voxel."""
    if origin_xyz_mm is None:
        return None

    # A flipped axis starts at the far end of the source axis it came from.
    source_index = np.zeros(3)
    for axis, source_axis in enumerate(permutation):
        if signs[axis] < 0:
            source_index[source_axis] = float(shape_zyx[source_axis] - 1)

    offset = np.zeros(3)
    for source_axis in range(3):
        step = source_index[source_axis] * float(spacing_zyx_mm[source_axis])
        offset += step * _axis_vector(matrix, source_axis)

    origin = np.asarray(origin_xyz_mm, dtype=np.float64) + offset
    return tuple(float(v) for v in origin)
