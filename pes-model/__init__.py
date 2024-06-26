bl_info = {
	"name": "PES .model format",
	"author": "foreground",
	"blender": (2, 79, 0),
	"category": "Import-Export",
	"version": (0, 3, 0),
}

import bpy
import bpy.props

from . import UI

def register():
	bpy.types.Mesh.pes_model_material = bpy.props.StringProperty(name = "Material")
	
	bpy.types.Armature.pes_model_simplified_skeleton = bpy.props.BoolProperty(name = ".model Simplified Skeleton", default = True, options = {'HIDDEN'})
	
	UI.register()

def unregister():
	UI.unregister()
