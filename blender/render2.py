import sys
import os
# sys.path.append('./')

import bpy

from .options import Options
from .load_bvh import load_bvh_allpath
from .scene import *
from .colormap import *
if __name__ == '__main__':
    args = Options(sys.argv).parse()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    path_list = []
    if args.bvh_path1 != "":
        path_list.append(args.bvh_path1)
    if args.bvh_path2 != "":
        path_list.append(args.bvh_path2)
    characters = load_bvh_allpath(path_list, multi=1.5)

    scene = make_scene(camera_position=camera_poses[args.camera_position])

    add_material_for_character(characters[0], mapping[args.color])
    add_material_for_character(characters[1], mapping[args.color2])

    bpy.ops.object.select_all(action='DESELECT')

    # music
    bpy.ops.object.speaker_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.sound.open_mono(filepath=args.music_path, relative_path=True)
    bpy.data.speakers["Speaker"].sound = bpy.data.sounds[0]
    bpy.data.sounds[0].copy()
    bpy.data.speakers["Speaker"].sound = bpy.data.sounds[1]

    add_rendering_parameters(bpy.context.scene, args, scene[1])

    if args.render:
        bpy.ops.render.render(animation=True, use_viewport=True)
