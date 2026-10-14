# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in repair for the missing OR node-graph icon in catalog 0.7.0/724f82e.

Requires usd-core (or the Isaac Sim USD runtime). Run with --help for usage.
The output must be a new sibling of the source so relative assets still resolve.
"""

import argparse
from pathlib import Path

ICON_ATTRIBUTE = (
    "/surgicalDrapes/surgicalDrapes_model/Looks/OmniSurface_bedSheet/"
    "file_texture.ui:nodegraph:node:icon"
)
BROKEN_ICON = (
    "./Collected_surgery_room_movie_with_heart_adjusted/"
    "surgicalDrapes_lookdev/core_definitions.file_texture.png"
)


def clear_broken_icon(layer):
    """Clear only the known bad icon default; preserve rendering inputs and metadata.

    Return whether the layer changed. Fail closed for a different scene or icon.
    The caller owns saving; this function only changes the supplied layer in memory.
    """
    from pxr import Sdf

    attribute = layer.GetAttributeAtPath(ICON_ATTRIBUTE)
    if attribute is None or attribute.typeName != Sdf.ValueTypeNames.Asset:
        raise ValueError(f"Expected asset attribute {ICON_ATTRIBUTE} in {layer.identifier}")
    if not attribute.HasDefaultValue():
        return False
    if attribute.default != Sdf.AssetPath(BROKEN_ICON):
        raise ValueError(f"Unexpected icon value; refusing to change {attribute.default!r}")
    attribute.ClearDefaultValue()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded Props/shared_OR_without_Mark/main.usd")
    parser.add_argument("output", type=Path, help="New sibling path, e.g. main.repaired.usd; never overwritten")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"Source does not exist: {source}")
    if source.parent != output.parent:
        parser.error("Output must be beside the source to preserve relative asset paths")
    if output.exists() or args.output.is_symlink():
        parser.error(f"Output already exists: {output}")
    if output.suffix not in {".usd", ".usda", ".usdc"}:
        parser.error("Output must have a .usd, .usda, or .usdc extension")

    try:
        from pxr import Sdf
    except ImportError:
        parser.error("USD is required: install usd-core or use Isaac Sim's Python runtime")

    # Work on an anonymous copy; never modify the downloaded source or its cache.
    layer = Sdf.Layer.OpenAsAnonymous(str(source))
    if layer is None:
        parser.error(f"Cannot open USD layer: {source}")
    try:
        changed = clear_broken_icon(layer)
    except ValueError as exc:
        parser.error(str(exc))
    # Reserve the output exclusively so an existing file cannot be overwritten.
    with output.open("xb"):
        pass
    try:
        if not layer.Export(str(output)):
            raise RuntimeError(f"Could not export repaired layer to {output}")
    except Exception:
        output.unlink()
        raise
    print(f"{'Cleared missing UI icon' if changed else 'Icon already cleared'}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
