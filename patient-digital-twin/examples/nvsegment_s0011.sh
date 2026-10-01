#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 /path/to/s0011/ct.nii.gz /path/to/NV-Segment-CTMR/NV-Segment-CTMR /new/output [--python /model/env/bin/python]" >&2
  exit 2
fi
ct="$1"
bundle="$2"
output="$3"
shift 3
exec python -m patient_digital_twin \
  --source nvsegment --input "$ct" --bundle-root "$bundle" \
  --classes aorta --format workflow --patient-id s0011 --output "$output" "$@"
