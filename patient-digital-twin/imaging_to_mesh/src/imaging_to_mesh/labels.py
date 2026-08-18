# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAISI/TotalSegmentator label groups used by the converter."""

from __future__ import annotations

CATEGORY_LABELS: dict[str, tuple[int, ...]] = {
    "Liver": (1,),
    "Spleen": (3,),
    "Pancreas": (4,),
    "Heart": (115,),
    "Body": (200,),
    "Gallbladder": (10,),
    "Stomach": (12,),
    "Small_bowel": (19,),
    "Colon": (62,),
    "Kidney": (5, 14),
    "Veins": (6, 7, 17, 58, 59, 60, 61, 109, 110, 111, 112, 113, 119, 123, 124, 125),
    "Lungs": (28, 29, 30, 31, 32),
    "Spine": (33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 97, 127, 131),
    "Ribs": tuple(range(63, 87)) + (114, 122),
    "Shoulders": (89, 90, 91, 92),
    "Hips": (95, 96),
    "Back_muscles": (98, 99, 100, 101, 102, 103, 104, 105, 106, 107),
}
