# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from vasculature_digital_twin import (
    HuToMuMapping,
    PreprocessingSettings,
    VolumePreprocessor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a CT volume into mu_volume + metadata cache artifacts.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", type=str, default=None, help="Path to a DICOM series directory.")
    src.add_argument("--nifti", type=str, default=None, help="Path to a NIfTI file (.nii / .nii.gz).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for cache outputs.")
    parser.add_argument("--hu-clip-min", type=float, default=PreprocessingSettings.hu_clip_min)
    parser.add_argument("--hu-clip-max", type=float, default=PreprocessingSettings.hu_clip_max)
    parser.add_argument(
        "--window-center",
        type=float,
        default=None,
        help="Level of the HU-to-mu ramp in HU. Requires --window-width.",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=None,
        help="Width of the HU-to-mu ramp in HU. Requires --window-center.",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=None,
        help=f"Attenuation coefficient (mm^-1) at the top of the ramp. Default: {HuToMuMapping.mu_max:g}.",
    )
    parser.add_argument(
        "--control-points",
        type=str,
        default=None,
        help=(
            "Piecewise-linear HU-to-mu curve as comma-separated hu:mu knots with strictly "
            "increasing HU, for independent slopes per HU band. Write it with an equals sign, "
            "as in --control-points=-1000:0,0:0.004,300:0.012,1500:0.02, so the leading minus "
            "is not read as another option. Cannot be combined with --window-center, "
            "--window-width or --mu-max."
        ),
    )
    parser.add_argument(
        "--reorient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reorient the volume into the canonical LPS frame using its direction cosines.",
    )
    parser.add_argument(
        "--save-hu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write raw HU as hu_volume.npy.",
    )
    return parser


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def _parse_control_points(text: str) -> tuple[tuple[float, float], ...]:
    """Parse ``hu:mu`` knots such as ``-1000:0,0:0.004,300:0.012,1500:0.02``."""
    knots: list[tuple[float, float]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        hu_text, separator, mu_text = token.partition(":")
        if not separator:
            raise SystemExit(f"--control-points expects hu:mu pairs, got {token!r}.")
        try:
            knots.append((float(hu_text), float(mu_text)))
        except ValueError:
            raise SystemExit(f"--control-points expects numeric hu:mu pairs, got {token!r}.") from None
    return tuple(knots)


def _build_mapping(args: argparse.Namespace) -> HuToMuMapping:
    if args.control_points is not None:
        conflicting = [
            name
            for name, value in (
                ("--window-center", args.window_center),
                ("--window-width", args.window_width),
                ("--mu-max", args.mu_max),
            )
            if value is not None
        ]
        if conflicting:
            raise SystemExit(
                "--control-points already fixes the whole curve, so it cannot be combined with "
                f"{', '.join(conflicting)}."
            )
        try:
            return HuToMuMapping(control_points=_parse_control_points(args.control_points))
        except ValueError as exc:
            raise SystemExit(f"--control-points is invalid: {exc}") from None

    if (args.window_center is None) != (args.window_width is None):
        raise SystemExit("--window-center and --window-width must be given together.")
    mu_max = HuToMuMapping.mu_max if args.mu_max is None else args.mu_max
    if args.window_center is None:
        return HuToMuMapping(mu_max=mu_max)
    return HuToMuMapping.from_window_level(
        window_center=args.window_center,
        window_width=args.window_width,
        mu_max=mu_max,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = PreprocessingSettings(
        hu_clip_min=args.hu_clip_min,
        hu_clip_max=args.hu_clip_max,
        hu_to_mu=_build_mapping(args),
    )
    dicom = _expand(args.dicom)
    nifti = _expand(args.nifti)
    output_dir = _expand(args.output_dir)

    if dicom is not None:
        preprocessor = VolumePreprocessor.from_dicom(dicom, settings=settings, reorient=args.reorient)
    else:
        preprocessor = VolumePreprocessor.from_nifti(nifti, settings=settings, reorient=args.reorient)

    mapping = settings.hu_to_mu
    if mapping.control_points is None:
        print(
            f"[vdt-preprocess-ct] HU-to-mu ramp: window_center={mapping.window_center:g} HU, "
            f"window_width={mapping.window_width:g} HU, mu in [{mapping.mu_min:g}, {mapping.mu_max:g}] mm^-1"
        )
    else:
        knots = " ".join(f"{hu:g}:{mu:g}" for hu, mu in mapping.points)
        print(f"[vdt-preprocess-ct] HU-to-mu piecewise curve ({len(mapping.points)} knots, HU:mu): {knots}")

    volume = preprocessor.preprocess(output_dir=output_dir)
    meta = volume.metadata
    if meta.anatomical_frame is None:
        print("[vdt-preprocess-ct] Orientation: unresolved (axes carry no anatomical meaning downstream)")
    else:
        print(
            f"[vdt-preprocess-ct] Orientation: source axes {meta.source_orientation} "
            f"-> {meta.anatomical_frame} canonical (axis 0 Superior, axis 1 Posterior, axis 2 Left)"
        )
    print(volume)

    if args.save_hu:
        hu_path = Path(output_dir) / "hu_volume.npy"
        np.save(hu_path, preprocessor.hu_volume_zyx.astype(np.float32))
        print(f"[vdt-preprocess-ct] Saved HU volume: {hu_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
