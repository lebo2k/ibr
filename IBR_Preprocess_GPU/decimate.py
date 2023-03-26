# note: it doesn't seem to be needed here, but in case try to do this manually through Blender instead, import/export should be done with X forward, Z up to match Agisoft's coordinate system

# Run with command:
# blender --background --python decimate.py -- <path_to_source.obj> <path_to_dest.obj> <decimation_ratio>
import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

DECIMATION_RATIO = float(argv[2])

# delete all default objects (camera, cube, lamp, etc.)
# seems even in background mode blender loads up default junk
for ob in bpy.data.objects:
	print("Deleting default object: " + ob.name)
	ob.select = True
bpy.ops.object.delete()

# import the obj
bpy.ops.import_scene.obj(filepath=argv[0])

# perform decimation on all imported objects
for ob in bpy.data.objects:
	mod = ob.modifiers.new (name='decimate', type='DECIMATE')
	mod.ratio = DECIMATION_RATIO

# write the output file
out = open(argv[1], "w")
bpy.ops.export_scene.obj(filepath=argv[1], axis_forward='-Z', axis_up='Y')
