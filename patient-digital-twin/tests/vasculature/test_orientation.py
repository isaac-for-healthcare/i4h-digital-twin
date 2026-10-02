# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical anatomical reorientation of CT volumes."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from vasculature_digital_twin import CANONICAL_FRAME, orientation_code, to_canonical_lps

# Patient-space (LPS) direction of each canonical array axis: axis 0 Superior,
# axis 1 Posterior, axis 2 Left.
_CANONICAL_AXIS_VECTORS = (
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
)

SPACING_ZYX_MM = (2.5, 0.7, 0.9)
ORIGIN_XYZ_MM = (-120.0, -140.0, 35.0)


@pytest.fixture
def canonical_hu() -> np.ndarray:
    """A volume with a distinct marker near the head, the back and the patient's left."""
    hu = np.full((8, 6, 4), -1000.0, dtype=np.float32)
    hu[7, 0, 0] = 300.0  # superior (head) end of axis 0
    hu[0, 5, 0] = 400.0  # posterior (back) end of axis 1
    hu[0, 0, 3] = 500.0  # left end of axis 2
    return hu


def direction_from_axes(axis_vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, ...]:
    """Build a row-major direction matrix from the patient direction of each array axis.

    Columns are index axes ``(i, j, k)``, i.e. array axes ``(2, 1, 0)``.
    """
    matrix = np.stack([axis_vectors[2], axis_vectors[1], axis_vectors[0]], axis=1)
    return tuple(float(v) for v in matrix.flatten())


def reorder(values: tuple, permutation: tuple[int, int, int]) -> tuple:
    return tuple(values[axis] for axis in permutation)


def as_stored(
    canonical: np.ndarray,
    permutation: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Re-express a canonical volume as a scanner would have stored it.

    Args:
        canonical: Volume in the canonical frame.
        permutation: Canonical axis that each stored axis corresponds to.
        signs: Whether each stored axis runs along (+1) or against (-1) that canonical axis.

    Returns:
        Tuple of (stored volume, direction cosines describing the stored volume).
    """
    stored = np.transpose(canonical, permutation)
    flips = tuple(axis for axis, sign in enumerate(signs) if sign < 0)
    if flips:
        stored = np.flip(stored, axis=flips)
    axis_vectors = tuple(signs[axis] * _CANONICAL_AXIS_VECTORS[permutation[axis]] for axis in range(3))
    return np.ascontiguousarray(stored), direction_from_axes(axis_vectors)


def voxel_position_mm(
    index_zyx: tuple[int, int, int],
    origin_xyz_mm: tuple[float, float, float],
    direction: tuple[float, ...],
    spacing_zyx_mm: tuple[float, float, float],
) -> np.ndarray:
    """Return the LPS position of a voxel, the quantity reorientation must preserve."""
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    position = np.asarray(origin_xyz_mm, dtype=np.float64)
    for axis, index in enumerate(index_zyx):
        position = position + index * spacing_zyx_mm[axis] * matrix[:, 2 - axis]
    return position


def marker_index(volume: np.ndarray, value: float) -> tuple[int, int, int]:
    return tuple(int(i) for i in np.argwhere(volume == value)[0])


class TestOrientationCode:
    """Reading the anatomical meaning of the array axes off a direction matrix."""

    def test_canonical_direction_is_spl(self):
        assert orientation_code(np.eye(3).flatten()) == "SPL"

    def test_first_slice_at_head_reads_as_inferior(self):
        _, direction = as_stored(np.zeros((2, 2, 2)), (0, 1, 2), (-1, 1, 1))
        assert orientation_code(direction) == "IPL"

    def test_ras_style_axes(self):
        _, direction = as_stored(np.zeros((2, 2, 2)), (0, 1, 2), (1, -1, -1))
        assert orientation_code(direction) == "SAR"

    def test_sagittal_acquisition(self):
        _, direction = as_stored(np.zeros((2, 2, 2)), (2, 0, 1), (1, 1, 1))
        assert orientation_code(direction) == "LSP"

    def test_unnormalised_direction_accepted(self):
        assert orientation_code((3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0)) == "SPL"


class TestCanonicalReorientation:
    """Volumes are permuted and flipped into the LPS frame without resampling."""

    def test_canonical_input_is_untouched(self, canonical_hu):
        result = to_canonical_lps(canonical_hu, np.eye(3).flatten(), SPACING_ZYX_MM, ORIGIN_XYZ_MM)

        assert result.is_identity
        assert result.source_code == "SPL"
        np.testing.assert_array_equal(result.hu_zyx, canonical_hu)
        assert result.spacing_zyx_mm == SPACING_ZYX_MM
        assert result.origin_xyz_mm == ORIGIN_XYZ_MM

    def test_unknown_direction_is_passed_through(self, canonical_hu):
        result = to_canonical_lps(canonical_hu, None, SPACING_ZYX_MM, ORIGIN_XYZ_MM)

        assert result.is_identity
        np.testing.assert_array_equal(result.hu_zyx, canonical_hu)

    def test_flipped_slice_order_puts_the_head_back_at_high_axis_0(self, canonical_hu):
        stored, direction = as_stored(canonical_hu, (0, 1, 2), (-1, 1, 1))
        assert marker_index(stored, 300.0)[0] == 0

        result = to_canonical_lps(stored, direction, SPACING_ZYX_MM, ORIGIN_XYZ_MM)

        assert result.flipped_axes == (0,)
        assert marker_index(result.hu_zyx, 300.0) == (7, 0, 0)

    def test_sagittal_acquisition_is_transposed(self, canonical_hu):
        # Stored axes run (Left, Superior, Posterior); shape follows the permutation.
        stored, direction = as_stored(canonical_hu, (2, 0, 1), (1, 1, 1))
        assert stored.shape == (4, 8, 6)

        result = to_canonical_lps(stored, direction, reorder(SPACING_ZYX_MM, (2, 0, 1)), ORIGIN_XYZ_MM)

        assert result.permutation == (1, 2, 0)
        np.testing.assert_array_equal(result.hu_zyx, canonical_hu)
        assert result.spacing_zyx_mm == SPACING_ZYX_MM

    @pytest.mark.parametrize(
        "permutation,signs",
        list(itertools.product(itertools.permutations((0, 1, 2)), itertools.product((1, -1), repeat=3))),
    )
    def test_every_axis_aligned_orientation_is_recovered(self, canonical_hu, permutation, signs):
        stored, direction = as_stored(canonical_hu, permutation, signs)
        result = to_canonical_lps(stored, direction, reorder(SPACING_ZYX_MM, permutation), ORIGIN_XYZ_MM)

        np.testing.assert_array_equal(result.hu_zyx, canonical_hu)
        assert result.spacing_zyx_mm == SPACING_ZYX_MM
        assert result.source_code == orientation_code(direction)
        assert orientation_code(result.direction) == "SPL"
        np.testing.assert_allclose(np.asarray(result.direction).reshape(3, 3), np.eye(3), atol=1e-12)
        assert result.max_obliquity_deg == pytest.approx(0.0)
        assert not result.is_oblique

    @pytest.mark.parametrize(
        "permutation,signs",
        list(itertools.product(itertools.permutations((0, 1, 2)), itertools.product((1, -1), repeat=3))),
    )
    def test_marker_positions_are_preserved(self, canonical_hu, permutation, signs):
        stored, direction = as_stored(canonical_hu, permutation, signs)
        stored_spacing = reorder(SPACING_ZYX_MM, permutation)
        result = to_canonical_lps(stored, direction, stored_spacing, ORIGIN_XYZ_MM)

        for value in (300.0, 400.0, 500.0):
            before = voxel_position_mm(marker_index(stored, value), ORIGIN_XYZ_MM, direction, stored_spacing)
            after = voxel_position_mm(
                marker_index(result.hu_zyx, value),
                result.origin_xyz_mm,
                result.direction,
                result.spacing_zyx_mm,
            )
            np.testing.assert_allclose(after, before, atol=1e-9)

    def test_output_is_contiguous(self, canonical_hu):
        stored, direction = as_stored(canonical_hu, (1, 2, 0), (-1, 1, -1))
        result = to_canonical_lps(stored, direction, reorder(SPACING_ZYX_MM, (1, 2, 0)), ORIGIN_XYZ_MM)
        assert result.hu_zyx.flags["C_CONTIGUOUS"]

    def test_unknown_origin_stays_unknown(self, canonical_hu):
        stored, direction = as_stored(canonical_hu, (0, 1, 2), (-1, 1, 1))
        assert to_canonical_lps(stored, direction, SPACING_ZYX_MM).origin_xyz_mm is None

    def test_summary_reports_the_applied_transform(self, canonical_hu):
        stored, direction = as_stored(canonical_hu, (0, 1, 2), (-1, 1, 1))
        summary = to_canonical_lps(stored, direction, SPACING_ZYX_MM, ORIGIN_XYZ_MM).summary()
        assert "IPL" in summary and CANONICAL_FRAME in summary


class TestObliqueAndInvalidInput:
    """Oblique acquisitions are flagged; unusable metadata is rejected."""

    def _rotated_about_superior(self, degrees: float) -> tuple[float, ...]:
        angle = np.radians(degrees)
        cos, sin = np.cos(angle), np.sin(angle)
        left = np.array([cos, sin, 0.0])
        posterior = np.array([-sin, cos, 0.0])
        return direction_from_axes((_CANONICAL_AXIS_VECTORS[0], posterior, left))

    def test_small_tilt_is_not_flagged(self, canonical_hu):
        result = to_canonical_lps(canonical_hu, self._rotated_about_superior(5.0), SPACING_ZYX_MM)
        assert result.max_obliquity_deg == pytest.approx(5.0, abs=1e-6)
        assert not result.is_oblique

    def test_large_tilt_is_flagged_and_nearest_axes_are_used(self, canonical_hu):
        result = to_canonical_lps(canonical_hu, self._rotated_about_superior(30.0), SPACING_ZYX_MM)

        assert result.max_obliquity_deg == pytest.approx(30.0, abs=1e-6)
        assert result.is_oblique
        assert result.is_identity
        np.testing.assert_array_equal(result.hu_zyx, canonical_hu)

    def test_residual_rotation_is_reported(self, canonical_hu):
        direction = self._rotated_about_superior(30.0)
        result = to_canonical_lps(canonical_hu, direction, SPACING_ZYX_MM)
        np.testing.assert_allclose(result.direction, direction, atol=1e-12)

    def test_degenerate_direction_rejected(self, canonical_hu):
        direction = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="degenerate column"):
            to_canonical_lps(canonical_hu, direction, SPACING_ZYX_MM)

    def test_non_finite_direction_rejected(self, canonical_hu):
        direction = (np.nan, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="non-finite"):
            to_canonical_lps(canonical_hu, direction, SPACING_ZYX_MM)

    def test_non_3d_volume_rejected(self):
        with pytest.raises(ValueError, match="Expected 3D volume"):
            to_canonical_lps(np.zeros((4, 4)), np.eye(3).flatten(), SPACING_ZYX_MM)


class TestAffineToLps:
    """NIfTI affines are RAS and must be converted before use."""

    def test_ras_identity_becomes_lps(self):
        from vasculature_digital_twin import affine_to_lps

        direction, spacing, origin = affine_to_lps(np.eye(4))

        np.testing.assert_allclose(np.asarray(direction).reshape(3, 3), np.diag([-1.0, -1.0, 1.0]))
        assert spacing == (1.0, 1.0, 1.0)
        assert origin == (0.0, 0.0, 0.0)

    def test_spacing_and_origin_are_extracted(self):
        from vasculature_digital_twin import affine_to_lps

        affine = np.diag([0.9, 0.7, 2.5, 1.0])
        affine[:3, 3] = [10.0, -20.0, 30.0]
        direction, spacing, origin = affine_to_lps(affine)

        assert spacing == pytest.approx((0.9, 0.7, 2.5))
        assert origin == pytest.approx((-10.0, 20.0, 30.0))
        assert orientation_code(direction) == "SAR"

    def test_bad_shape_rejected(self):
        from vasculature_digital_twin import affine_to_lps

        with pytest.raises(ValueError, match="4x4 affine"):
            affine_to_lps(np.eye(3))


class TestLoaderIntegration:
    """End-to-end: a NIfTI in RAS lands in the canonical frame with metadata recorded."""

    def _write_nifti(self, path: Path, canonical_hu: np.ndarray) -> tuple[float, float, float]:
        nib = pytest.importorskip("nibabel")

        # Store the volume RAS-style, the layout nibabel-authored files usually carry.
        stored, _ = as_stored(canonical_hu, (0, 1, 2), (1, -1, -1))
        spacing_ijk = (SPACING_ZYX_MM[2], SPACING_ZYX_MM[1], SPACING_ZYX_MM[0])
        affine = np.diag([*spacing_ijk, 1.0])
        affine[:3, 3] = [10.0, -20.0, 30.0]
        nib.save(nib.Nifti1Image(np.transpose(stored, (2, 1, 0)), affine), path)
        return spacing_ijk

    def test_nifti_is_reoriented_and_annotated(self, tmp_path: Path, canonical_hu):
        from vasculature_digital_twin.ct.dicom_ingest import load_nifti_hu

        path = tmp_path / "ct.nii.gz"
        self._write_nifti(path, canonical_hu)

        ct = load_nifti_hu(path)

        np.testing.assert_allclose(ct.hu_zyx, canonical_hu)
        assert ct.spacing_zyx_mm == pytest.approx(SPACING_ZYX_MM)
        assert ct.anatomical_frame == CANONICAL_FRAME
        assert ct.source_orientation == "SAR"

    def test_reorient_disabled_keeps_stored_layout(self, tmp_path: Path, canonical_hu):
        from vasculature_digital_twin.ct.dicom_ingest import load_nifti_hu

        path = tmp_path / "ct.nii.gz"
        self._write_nifti(path, canonical_hu)
        stored, _ = as_stored(canonical_hu, (0, 1, 2), (1, -1, -1))

        ct = load_nifti_hu(path, reorient=False)

        np.testing.assert_allclose(ct.hu_zyx, stored)
        assert ct.anatomical_frame is None

    def test_metadata_carries_the_frame_downstream(self, tmp_path: Path, canonical_hu):
        from vasculature_digital_twin import PreprocessedVolume, VolumePreprocessor

        path = tmp_path / "ct.nii.gz"
        self._write_nifti(path, canonical_hu)
        cache = tmp_path / "ct_cache"

        VolumePreprocessor.from_nifti(path).preprocess(output_dir=cache)
        metadata = PreprocessedVolume.load(cache).metadata

        assert metadata.anatomical_frame == CANONICAL_FRAME
        assert metadata.source_orientation == "SAR"
        np.testing.assert_allclose(np.asarray(metadata.direction).reshape(3, 3), np.eye(3), atol=1e-12)

    def test_oblique_acquisition_warns(self, tmp_path: Path, canonical_hu):
        nib = pytest.importorskip("nibabel")
        from vasculature_digital_twin.ct.dicom_ingest import load_nifti_hu

        angle = np.radians(25.0)
        affine = np.eye(4)
        affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        path = tmp_path / "oblique.nii.gz"
        nib.save(nib.Nifti1Image(np.transpose(canonical_hu, (2, 1, 0)), affine), path)

        with pytest.warns(UserWarning, match="oblique acquisition"):
            load_nifti_hu(path)

    def test_bare_numpy_volume_has_no_frame(self, canonical_hu):
        from vasculature_digital_twin import VolumePreprocessor

        metadata = VolumePreprocessor.from_numpy(canonical_hu).preprocess().metadata
        assert metadata.anatomical_frame is None
