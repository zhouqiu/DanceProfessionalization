import bpy
import sys
# sys.path.append('./')


def add_floor(size, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, location=location)
    floor = bpy.context.object
    floor.name = 'floor'

    floor_mat = bpy.data.materials.new(name="floorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_text = floor_mat.node_tree.nodes.new("ShaderNodeTexChecker")
    floor_text.inputs[3].default_value = 150
    floor_mat.node_tree.links.new(bsdf.inputs['Base Color'], floor_text.outputs['Color'])

    floor.data.materials.append(floor_mat)
    return floor


def add_camera(location, rotation):
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
    camera = bpy.context.object
    return camera


def add_light(location):
    bpy.ops.object.light_add(type='SUN', location=location)
    sun = bpy.context.object
    sun.data.energy=4.5
    return sun


def make_scene(floor_size=2000,    camera_position=(60, 0, 19), camera_rotation=(1.358, 0, 1.54),
               light_position=(5, 0, 20)):
    floor = add_floor(floor_size, (0,0,0))

    camera = add_camera(camera_position, camera_rotation)
    light = add_light(light_position)
    bpy.ops.object.select_all(action='DESELECT')
    floor.select_set(True)
    camera.select_set(True)
    light.select_set(True)
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="Scene")
    bpy.ops.object.select_all(action='DESELECT')
    return [floor, camera, light]


def add_rendering_parameters(scene, args, camera):
    scene.render.resolution_x = args.resX
    scene.render.resolution_y = args.resY
    scene.frame_start = args.frame_start
    scene.frame_end = args.frame_end
    scene.camera = camera
    scene.render.filepath = args.save_path

    if args.render_engine == 'cycles':
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
    elif args.render_engine == 'eevee':
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'  #
    bpy.context.scene.render.ffmpeg.audio_codec = 'MP3'

    return scene


def add_material_for_character(objs,color):
    char_mat = bpy.data.materials.new(name="characterMaterial")
    char_mat.use_nodes = True
    bsdf = char_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = color   # character material color
    for obj in objs:
        obj.data.materials.append(char_mat)
