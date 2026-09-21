# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct a patient anatomy digital twin."""

from pathlib import Path

from patient_digital_twin import AnatomicalStructure, HumanBody, Kind, System

body = HumanBody(
    {
        name: AnatomicalStructure(name, Kind.ORGAN)
        for name in ("liver", "kidney_right", "kidney_left")
    }
)
print([structure.name for structure in body.select(system=System.URINARY)])

body.configure_anatomy(Path(__file__).with_name("anatomy.yaml"))
body.set_anatomy_enabled(False)
assert all(structure.is_empty for structure in body.structures.values())
body.set_anatomy_enabled(True)
body.system("urinary").set_enabled(False)
assert body.system("urinary").is_empty
# Structures without geometry stay empty even when enabled: no mesh was fabricated.
print("Liver enabled:", body.structures["liver"].enabled)
print("Liver has no segmented mesh:", body.structures["liver"].is_empty)
