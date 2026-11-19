# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native CT/scan volume helper, owned by ``xray_simulator.scan_volume``.

This module used to be a byte-identical copy of the sensor-simulation helper. It now
re-exports the single implementation so conversion, artifact, and replay behavior
cannot drift between the two packages.
"""

from xray_simulator.scan_volume import (
    FORMAT_VERSION,
    Conversion,
    ScanVolume,
    affine,
    convert,
    digest,
    export_ct,
    from_array,
    from_dicom,
    from_nifti,
    implementation,
    load_artifact,
    read_ct,
    replay,
)

__all__ = [
    "FORMAT_VERSION",
    "Conversion",
    "ScanVolume",
    "affine",
    "convert",
    "digest",
    "export_ct",
    "from_array",
    "from_dicom",
    "from_nifti",
    "implementation",
    "load_artifact",
    "read_ct",
    "replay",
]
