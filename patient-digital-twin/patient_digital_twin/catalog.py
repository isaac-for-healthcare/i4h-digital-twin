# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curated kinds, system membership, and temporary matching hints.

Application metadata, not a clinical reference ontology.
"""

from __future__ import annotations

from .structures import (
    Kind,
    System,
)


def _catalog(field="kind") -> dict:
    """Build kinds, system membership, or temporary joint-matching hints."""
    result = {}

    def add(name, kind, systems, regions, *, side=None):
        result[name] = {
            "kind": kind,
            "systems": frozenset(systems),
            "matching": (frozenset(regions.split()), side),
        }[field]

    def paired(base, kind, systems, regions):
        for side in ("left", "right"):
            add(f"{base}_{side}", kind, systems, regions, side=side)

    S, K = System, Kind
    for name in ("liver", "gallbladder", "stomach", "duodenum"):
        add(name, K.ORGAN, [S.DIGESTIVE], "abdomen")
    for name in ("small_bowel", "colon"):
        add(name, K.ORGAN, [S.DIGESTIVE], "abdomen pelvis")
    add("esophagus", K.ORGAN, [S.DIGESTIVE], "neck thorax abdomen")
    add("pancreas", K.ORGAN, [S.DIGESTIVE, S.ENDOCRINE], "abdomen")
    add("spleen", K.ORGAN, [S.LYMPHATIC_IMMUNE], "abdomen")
    paired("kidney", K.ORGAN, [S.URINARY, S.ENDOCRINE], "abdomen")
    paired("adrenal_gland", K.ORGAN, [S.ENDOCRINE], "abdomen")
    add("thyroid_gland", K.ORGAN, [S.ENDOCRINE], "neck")
    add("urinary_bladder", K.ORGAN, [S.URINARY], "pelvis")
    add("prostate", K.ORGAN, [S.REPRODUCTIVE], "pelvis")
    add("trachea", K.AIRWAY, [S.RESPIRATORY], "neck thorax")
    add("brain", K.ORGAN, [S.NERVOUS], "head")
    add("spinal_cord", K.ORGAN, [S.NERVOUS], "neck thorax abdomen")
    add("heart", K.ORGAN, [S.CARDIOVASCULAR], "thorax")
    add(
        "atrial_appendage_left",
        K.ORGAN_PART,
        [S.CARDIOVASCULAR],
        "thorax",
        side="left",
    )
    paired("lung", K.ORGAN, [S.RESPIRATORY], "thorax")
    for side in ("left", "right"):
        lobes = ("upper", "lower") if side == "left" else ("upper", "middle", "lower")
        for lobe in lobes:
            add(
                f"lung_{lobe}_lobe_{side}",
                K.ORGAN_PART,
                [S.RESPIRATORY],
                "thorax",
                side=side,
            )
        add(
            f"kidney_cyst_{side}",
            K.FINDING,
            [S.URINARY],
            "abdomen",
            side=side,
        )

    for name, regions in {
        "aorta": "thorax abdomen",
        "pulmonary_vein": "thorax",
        "brachiocephalic_trunk": "thorax neck",
        "superior_vena_cava": "thorax",
        "inferior_vena_cava": "abdomen thorax",
    }.items():
        add(name, K.VESSEL, [S.CARDIOVASCULAR], regions)
    # This segmentation label explicitly merges two different veins.
    add("portal_vein_and_splenic_vein", K.GROUP, [S.CARDIOVASCULAR], "abdomen")
    for base, regions in {
        "subclavian_artery": "thorax neck",
        "common_carotid_artery": "neck thorax",
        "brachiocephalic_vein": "thorax",
        "iliac_artery": "abdomen pelvis",
        "iliac_vena": "abdomen pelvis",
    }.items():
        paired(base, K.VESSEL, [S.CARDIOVASCULAR], regions)

    for base, regions in {
        "humerus": "upper_limb",
        "scapula": "upper_limb thorax",
        "clavicula": "upper_limb thorax",
        "femur": "lower_limb",
        "hip": "pelvis",
    }.items():
        # TotalSegmentator's 'hip' label denotes the hip bone, not the joint.
        paired(base, K.BONE, [S.SKELETAL], regions)
    add("sacrum", K.BONE, [S.SKELETAL], "pelvis")
    add("sternum", K.BONE, [S.SKELETAL], "thorax")
    add("skull", K.GROUP, [S.SKELETAL], "head")
    add("vertebrae", K.GROUP, [S.SKELETAL], "neck thorax abdomen pelvis")
    add("intervertebral_discs", K.GROUP, [S.SKELETAL], "neck thorax abdomen pelvis")
    add("costal_cartilages", K.GROUP, [S.SKELETAL], "thorax")
    for section, count, region in (
        ("C", 7, "neck"),
        ("T", 12, "thorax"),
        ("L", 6, "abdomen"),
        ("S", 1, "pelvis"),
    ):
        for number in range(1, count + 1):
            add(
                f"vertebrae_{section}{number}",
                K.BONE,
                [S.SKELETAL],
                region,
            )
    for side in ("left", "right"):
        for number in range(1, 13):
            add(f"rib_{side}_{number}", K.BONE, [S.SKELETAL], "thorax", side=side)
    for base in ("gluteus_maximus", "gluteus_medius", "gluteus_minimus"):
        paired(base, K.MUSCLE, [S.MUSCULAR], "pelvis lower_limb")
    paired("autochthon", K.GROUP, [S.MUSCULAR], "thorax abdomen pelvis")
    paired("iliopsoas", K.GROUP, [S.MUSCULAR], "abdomen pelvis lower_limb")
    return result


CATALOG = _catalog()
SYSTEM_MEMBERSHIP = _catalog("systems")


def matching_hints(name):
    """Temporary regions/laterality for matching; never stored on a structure."""
    return _catalog("matching").get(name, (frozenset(), None))


def structure_systems(name):
    """Look up system membership by canonical name, independent of mesh state."""
    return SYSTEM_MEMBERSHIP.get(name, frozenset())
