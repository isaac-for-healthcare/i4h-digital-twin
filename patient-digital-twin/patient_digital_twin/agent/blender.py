# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small scene handoff, executed inside Blender through upstream Blender MCP."""
import contextlib
import json
from pathlib import Path


def open_asset(directory):
    import bpy

    folder = Path(directory)
    settings = json.loads((folder / 'edit.json').read_text())
    scene = bpy.data.scenes.new('I4H ' + settings['token'])
    bpy.context.window.scene = scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    bpy.ops.wm.usd_import(filepath=str(folder / 'input.usdc'), import_visible_only=False,
                          property_import_mode='ALL', merge_parent_xform=False)
    from mathutils import Matrix

    # Bake import unit/parent transforms so mesh editing coordinates are meters.
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.data.transform(obj.matrix_world)
            obj.parent = None
            obj.matrix_world = Matrix.Identity(4)
    scene['i4h_session'] = settings['token']
    (folder / 'opened').touch()


def export_asset(directory):
    import bpy

    folder = Path(directory)
    settings = json.loads((folder / 'edit.json').read_text())
    scene = bpy.context.scene
    if scene.get('i4h_session') != settings['token']:
        raise ValueError('Return to the I4H editing scene before exporting')
    # Preserve the agent result before export/validation so failed runs can be recovered.
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'edited.blend'), copy=True)
    meshes = [o for o in scene.objects if o.type == 'MESH']
    paths = [o.data.get('i4h_path') for o in meshes]
    if len(paths) != len(set(paths)) or set(paths) != set(settings['paths']):
        raise ValueError('Keep the existing meshes and their i4h_path identity; add/delete/split/join is unsupported')
    if any(o.mode != 'OBJECT' for o in meshes):
        raise ValueError('Leave Edit Mode before exporting')
    if any(o.type not in {'MESH', 'EMPTY', 'LIGHT', 'CAMERA'} for o in scene.objects):
        raise ValueError('Only mesh edits, materials and textures are supported')
    for obj in meshes:
        obj['i4h_path'] = obj.data['i4h_path']
        obj['i4h_hidden'] = obj.hide_render or not obj.visible_get()
    with _export_all_meshes(scene):
        bpy.ops.wm.usd_export(filepath=str(folder / 'edited.usdc'), export_animation=False,
                              export_materials=True, generate_preview_surface=True,
                              export_custom_properties=True, triangulate_meshes=True,
                              export_lights=False, export_cameras=False, export_hair=False,
                              export_armatures=False, export_shapekeys=False,
                              relative_paths=True, overwrite_textures=True)


@contextlib.contextmanager
def _export_all_meshes(scene):
    """USD omits hidden geometry; reveal it temporarily and carry visibility as metadata."""
    import bpy

    saved, hidden = [], []

    def reveal(owner, attribute):
        saved.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, False)

    try:
        layers = [bpy.context.view_layer.layer_collection]
        while layers:
            layer = layers.pop()
            layers.extend(layer.children)
            reveal(layer, 'exclude')
            reveal(layer, 'hide_viewport')
            reveal(layer.collection, 'hide_viewport')
            reveal(layer.collection, 'hide_render')
        for obj in scene.objects:
            hidden.append((obj, obj.hide_get()))
            obj.hide_set(False)
            reveal(obj, 'hide_viewport')
            reveal(obj, 'hide_render')
        bpy.context.view_layer.update()
        yield
    finally:
        for obj, value in hidden:
            obj.hide_set(value)
        for owner, attribute, value in reversed(saved):
            setattr(owner, attribute, value)
