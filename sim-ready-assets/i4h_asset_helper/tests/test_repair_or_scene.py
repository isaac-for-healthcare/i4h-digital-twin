# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD-backed regression tests; install usd-core to run them outside Isaac Sim."""

import pytest

from i4h_asset_helper.repair_or_scene import BROKEN_ICON, ICON_ATTRIBUTE, clear_broken_icon, main

pytest.importorskip("pxr")
from pxr import Sdf, UsdUtils  # noqa: E402


def make_scene(path):
    layer = Sdf.Layer.CreateNew(str(path))
    prim = Sdf.CreatePrimInLayer(layer, Sdf.Path(ICON_ATTRIBUTE).GetPrimPath())
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "NodeGraph"
    icon = Sdf.AttributeSpec(prim, "ui:nodegraph:node:icon", Sdf.ValueTypeNames.Asset)
    icon.default = Sdf.AssetPath(BROKEN_ICON)
    texture = Sdf.AttributeSpec(prim, "inputs:texture", Sdf.ValueTypeNames.Asset)
    texture.default = Sdf.AssetPath("./real-texture.png")
    texture.connectionPathList.explicitItems = [Sdf.Path("/Material.inputs:texture")]
    (path.parent / "real-texture.png").write_bytes(b"texture dependency")
    layer.Save()
    return layer


def test_dependency_removed_without_changing_rendering_or_source(tmp_path):
    source = tmp_path / "main.usda"
    layer = make_scene(source)
    original_bytes = source.read_bytes()
    original_text = layer.ExportToString()
    _, _, before = UsdUtils.ComputeAllDependencies(str(source))
    assert len(before) == 1
    assert before[0].endswith("core_definitions.file_texture.png")

    output = tmp_path / "main.repaired.usda"
    assert main([str(source), str(output)]) == 0
    assert source.read_bytes() == original_bytes
    repaired = Sdf.Layer.FindOrOpen(str(output))
    # The full serialized layer differs by exactly the icon default, including
    # preservation of texture inputs, connections, metadata, and prim structure.
    assert repaired.ExportToString() == original_text.replace(f" = @{BROKEN_ICON}@", "")
    _, assets, after = UsdUtils.ComputeAllDependencies(str(output))
    assert not after
    assert str(tmp_path / "real-texture.png") in assets
    assert clear_broken_icon(repaired) is False


@pytest.mark.parametrize("value", ["./valid-icon.png", "./other/core_definitions.file_texture.png"])
def test_refuses_unexpected_icon(tmp_path, value):
    layer = make_scene(tmp_path / "main.usda")
    layer.GetAttributeAtPath(ICON_ATTRIBUTE).default = Sdf.AssetPath(value)
    before = layer.ExportToString()
    with pytest.raises(ValueError, match="Unexpected icon"):
        clear_broken_icon(layer)
    assert layer.ExportToString() == before


def test_refuses_unrelated_layer():
    with pytest.raises(ValueError, match="Expected asset attribute"):
        clear_broken_icon(Sdf.Layer.CreateAnonymous())


@pytest.mark.parametrize("target", ["main.usda", "existing.usda", "elsewhere/fixed.usda", "fixed.txt"])
def test_cli_refuses_unsafe_output(tmp_path, target):
    source = tmp_path / "main.usda"
    make_scene(source)
    existing = tmp_path / "existing.usda"
    existing.write_text("keep me")
    before = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(tmp_path / target)])
    assert exc.value.code == 2
    assert source.read_bytes() == before
    assert existing.read_text() == "keep me"
