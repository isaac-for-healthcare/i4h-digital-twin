# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary-mask surface extraction bundled with patient-digital-twin."""

from .mesh import mask_to_mesh

__all__ = ["mask_to_mesh"]
