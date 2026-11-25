# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default vessel wall stiffness, one Young's modulus per segmented vessel.

A patient bundle gives every exported vessel its own physics region (see
``export.py``). Each region carries the Young's modulus below, and the physics
simulation turns it into solver stiffness. To support a new segmentation model's
vessel, add a row: its names also join ``anatomy.CATALOG`` as vessels, so
importers keep those labels.

Arterial values are Wang & Parker 2004, Table 1 (the Westerhof 1969 arterial
tree): young adults at about 100 mmHg. Venous values are derived from the
reference wave speeds c0 in Mueller & Toro 2014, Table 8, as E = 2 rho c0^2 R / h
with rho = 1050 kg/m^3 and an assumed wall h = 0.1 R, at about 5 mmHg supine.
Pulmonary values are derived from Qureshi et al. 2014 (E h / r0 = 195 mmHg) with
the same h = 0.1 R. Stiffness rises with age and with pressure, steeply for veins.
"""

from __future__ import annotations

import re
from typing import NamedTuple

SOURCES = {
    "wang_parker_2004": "Wang JJ, Parker KH. Wave propagation in a model of the arterial circulation. "
    "J Biomech 2004;37:457-470, Table 1. doi:10.1016/j.jbiomech.2003.09.007",
    "muller_toro_2014": "Mueller LO, Toro EF. A global multiscale mathematical model for the human circulation "
    "with emphasis on the venous system. International Journal for Numerical Methods in Biomedical "
    "Engineering 2014;30:681-725, Table 8. "
    "doi:10.1002/cnm.2622",
    "qureshi_2014": "Qureshi MU et al. Numerical simulation of blood flow and pressure drop in the pulmonary "
    "arterial and venous circulation. Biomech Model Mechanobiol 2014;13:1137-1154, section 2.4. "
    "doi:10.1007/s10237-014-0563-y",
}


class VesselStiffness(NamedTuple):
    """Wall Young's modulus in kPa and the ``SOURCES`` key it comes from."""

    young_modulus_kpa: float
    source: str


ARTERY_DEFAULT = VesselStiffness(800.0, "wang_parker_2004")  # Medium arteries; unlisted arteries.
VEIN_DEFAULT = VesselStiffness(50.0, "muller_toro_2014")  # c0 of about 1.5 m/s; unlisted veins.

# Space-separated names; '{s}' expands to left and right.
_ROWS = [
    ("aorta ascending_aorta aortic_arch thoracic_aorta abdominal_aorta brachiocephalic_trunk "
     "common_carotid_artery_{s} subclavian_artery_{s} renal_artery_{s} iliac_artery_{s} "
     "common_iliac_artery_{s}", 400.0, "wang_parker_2004"),
    ("internal_carotid_artery_{s} external_carotid_artery_{s} vertebral_artery_{s} "
     "external_iliac_artery_{s} femoral_artery_{s} popliteal_artery_{s}", 800.0, "wang_parker_2004"),
    ("internal_iliac_artery_{s}", 1600.0, "wang_parker_2004"),
    ("superior_vena_cava", 21.0, "muller_toro_2014"),  # c0 1.000 m/s
    ("inferior_vena_cava", 23.0, "muller_toro_2014"),  # c0 1.053 m/s
    ("brachiocephalic_vein_{s}", 40.0, "muller_toro_2014"),  # c0 1.36 (right) and 1.41 (left) m/s
    ("iliac_vena_{s}", 38.0, "muller_toro_2014"),  # common iliac vein, c0 1.342 m/s
    ("femoral_vein_{s}", 66.0, "muller_toro_2014"),  # c0 1.775 m/s
    ("pulmonary_artery pulmonary_vein", 260.0, "qureshi_2014"),
    ("portal_vein_and_splenic_vein", VEIN_DEFAULT.young_modulus_kpa, VEIN_DEFAULT.source),
]

VESSEL_STIFFNESS: dict[str, VesselStiffness] = {
    name.replace("{s}", side): VesselStiffness(value, source)
    for names, value, source in _ROWS
    for name in names.split()
    for side in (("left", "right") if "{s}" in name else ("",))
}


def is_vein(name: str) -> bool:
    """True for venous names (containing ``vein`` or ``vena``); every other vessel counts as an artery."""
    return bool(re.search(r"vein|vena", name))


def circulation(name: str) -> str:
    """The circulation a vessel belongs to; gap bridging joins only vessels of one circulation.

    ``pulmonary_artery``, ``pulmonary_vein``, ``portal`` (portal, splenic and mesenteric
    veins), ``systemic_vein`` or ``systemic_artery``.
    """
    if "pulmonary" in name:
        return "pulmonary_vein" if is_vein(name) else "pulmonary_artery"
    if re.search(r"portal|splenic_vein|mesenteric_vein", name):
        return "portal"
    return "systemic_vein" if is_vein(name) else "systemic_artery"


def vessel_stiffness(name: str) -> VesselStiffness:
    """Stiffness for a canonical vessel name; unlisted veins and arteries get the class default.

    Example:
        ``vessel_stiffness("femoral_artery_left").young_modulus_kpa == 800.0``
    """
    if name in VESSEL_STIFFNESS:
        return VESSEL_STIFFNESS[name]
    return VEIN_DEFAULT if is_vein(name) else ARTERY_DEFAULT
