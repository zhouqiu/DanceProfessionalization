
# blender one file
blender --background --enable-autoexec -P render1.py -- --music_path musicPath --bvh_path1 bvhPath --color blue --camera_position near --frame_end xxx --render --save_path savePath
# blender two files
blender --background --enable-autoexec -P render2.py -- --music_path musicPath --bvh_path1 bvhPath1 --color red --bvh_path2 bvhPath2 --color2 blue --camera_position far --frame_end xxx --render --save_path savePath




