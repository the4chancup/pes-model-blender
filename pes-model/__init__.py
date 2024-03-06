bl_info = {
	"name": "PES .model format",
	"author": "foreground",
	"blender": (2, 79, 0),
	"category": "Import-Export",
	"version": (0, 2, 3),
}

import bpy
import bpy.props

from . import UI

def register():
	bpy.types.Mesh.pes_model_material = bpy.props.StringProperty(name = "Material")
	UI.register()

def unregister():
	UI.unregister()
