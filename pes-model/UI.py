import bpy
import bpy.props
import bpy_extras.io_utils

from . import ModelFile, IO, Skeleton



class PES_Model_Scene_Import(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
	"""Load a PES model file"""
	bl_idname = "import_scene.pes_model"
	bl_label = "Import .model"
	bl_options = {'REGISTER', 'UNDO'}
	
	extensions_enabled = bpy.props.BoolProperty(name = "Enable blender-pes-model extensions", default = True)
	mesh_names = bpy.props.BoolProperty(name = "Store mesh names", default = True)
	loop_preservation = bpy.props.BoolProperty(name = "Preserve split vertices", default = True)
	mesh_splitting = bpy.props.BoolProperty(name = "Autosplit overlarge meshes", default = True)
	ignore_parser_warnings = bpy.props.BoolProperty(name = "Ignore parser warnings", default = False)
	lenient_parsing = bpy.props.BoolProperty(name = "Fix corruptions where possible", default = True)
	skeleton_simplification = bpy.props.BoolProperty(name = "Import simplified skeletons if possible", default = True)
	
	import_label = "PES model (.model)"
	
	filename_ext = ".model"
	filter_glob = bpy.props.StringProperty(default="*.model", options={'HIDDEN'})
	
	def invoke(self, context, event):
		self.extensions_enabled = context.scene.pes_model_import_extensions_enabled
		self.mesh_names = context.scene.pes_model_import_mesh_names
		self.loop_preservation = context.scene.pes_model_import_loop_preservation
		self.mesh_splitting = context.scene.pes_model_import_mesh_splitting
		self.skeleton_simplification = context.scene.pes_model_import_skeleton_simplification
		self.ignore_parser_warnings = context.scene.pes_model_ignore_parser_warnings
		self.lenient_parsing = context.scene.pes_model_lenient_parsing
		return bpy_extras.io_utils.ImportHelper.invoke(self, context, event)
	
	def execute(self, context):
		filename = self.filepath
		
		parserSettings = ModelFile.ParserSettings()
		parserSettings.ignoreParserWarnings = self.ignore_parser_warnings
		parserSettings.strictParsing = not self.lenient_parsing
		
		importSettings = IO.ImportSettings()
		importSettings.enableExtensions = self.extensions_enabled
		importSettings.enableMeshNames = self.mesh_names
		importSettings.enableVertexLoopPreservation = self.loop_preservation
		importSettings.enableMeshSplitting = self.mesh_splitting
		importSettings.enableSkeletonSimplification = self.skeleton_simplification
		
		try:
			(modelFile, warnings) = ModelFile.readModelFile(filename, parserSettings)
		except ModelFile.FatalParserWarning as e:
			errorMessage = (
'''ERROR: Unexpected .model property: %s

This file uses features of the .model file format the plugin does not understand. Editing it may not work correctly.
The plugin can ignore this problem, continue, and hope for the best. To try this, enable the Ignore Parser Warnings import setting.''' % str(e)
			)
			self.report({'ERROR'}, errorMessage)
			return {'CANCELLED'}
		
		if len(warnings) > 0:
			warningString = "        \t".join(warnings)
			warningMessage = (
'''WARNING

This file uses features of the .model file format the plugin does not understand. Editing it may not work correctly.
Whatever has made it into blender will probably work fine; but if you export this to replace a PES file, it will probably not work as expected.
Here be dragons, test carefully, etc.

Specifically, the following warnings are reported:
        %s

If you get this file to work, let me know, because that's good information.''' % warningString
			)
			self.report({'WARNING'}, warningMessage)
		
		IO.importModel(context, modelFile, filename, importSettings)
		
		return {'FINISHED'}

class PES_Model_Scene_Export_Object(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
	"""Export an individual object as a PES .model file"""
	bl_idname = "export_scene.pes_model_object"
	bl_label = "Export .model"
	bl_options = {'REGISTER'}
	
	objectName = bpy.props.StringProperty("Object to export")
	extensions_enabled = bpy.props.BoolProperty(name = "Enable blender-pes-model extensions", default = True)
	mesh_names = bpy.props.BoolProperty(name = "Store mesh names", default = True)
	loop_preservation = bpy.props.BoolProperty(name = "Preserve split vertices", default = True)
	mesh_splitting = bpy.props.BoolProperty(name = "Autosplit overlarge meshes", default = True)
	
	export_label = "PES model (.model)"
	
	filename_ext = ".model"
	filter_glob = bpy.props.StringProperty(default="*.model", options={'HIDDEN'})
	
	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT' and context.active_object != None
	
	def invoke(self, context, event):
		self.objectName = context.active_object.name
		self.extensions_enabled = context.active_object.pes_model_export_extensions_enabled
		self.mesh_names = context.active_object.pes_model_export_mesh_names
		self.loop_preservation = context.active_object.pes_model_export_loop_preservation
		self.mesh_splitting = context.active_object.pes_model_export_mesh_splitting
		if context.active_object.pes_model_filename != "":
			self.filepath = context.active_object.pes_model_filename
		return bpy_extras.io_utils.ExportHelper.invoke(self, context, event)
	
	def execute(self, context):
		exportSettings = IO.ExportSettings()
		exportSettings.enableExtensions = self.extensions_enabled
		exportSettings.enableMeshNames = self.mesh_names
		exportSettings.enableVertexLoopPreservation = self.loop_preservation
		exportSettings.enableMeshSplitting = self.mesh_splitting
		
		try:
			modelFile = IO.exportModel(context, self.objectName, exportSettings)
		except IO.ExportError as error:
			self.report({'ERROR'}, "Error exporting .model: " + "; ".join(error.errors))
			print("Error exporting .model:\n" + "\n".join(error.errors))
			return {'CANCELLED'}
		
		ModelFile.writeModelFile(modelFile, self.filepath)
		
		self.report({'INFO'}, "Model exported successfully.") 
		
		return {'FINISHED'}


def PES_Model_Scene_Import_MenuItem(self, context):
	self.layout.operator(PES_Model_Scene_Import.bl_idname, text=PES_Model_Scene_Import.import_label)

class PES_Model_Scene_Panel_Model_Import_Settings(bpy.types.Menu):
	"""Import Settings"""
	bl_label = "Import settings"
	
	def draw(self, context):
		self.layout.prop(context.scene, 'pes_model_import_extensions_enabled')
		
		row = self.layout.row()
		row.prop(context.scene, 'pes_model_import_mesh_names')
		row.enabled = context.scene.pes_model_import_extensions_enabled
		
		row = self.layout.row()
		row.prop(context.scene, 'pes_model_import_loop_preservation')
		row.enabled = context.scene.pes_model_import_extensions_enabled
		
		row = self.layout.row()
		row.prop(context.scene, 'pes_model_import_mesh_splitting')
		row.enabled = context.scene.pes_model_import_extensions_enabled
		
		row = self.layout.row()
		row.prop(context.scene, 'pes_model_import_skeleton_simplification')
		
		self.layout.prop(context.scene, 'pes_model_ignore_parser_warnings')
		
		self.layout.prop(context.scene, 'pes_model_lenient_parsing')

class PES_Model_Scene_Panel_Model_Remove(bpy.types.Operator):
	"""Disable separate exporting"""
	bl_idname = "pes_model.remove_exportable"
	bl_label = "Remove"
	bl_options = {'UNDO', 'INTERNAL'}
	
	objectName = bpy.props.StringProperty(name = "Object to remove")
	
	def execute(self, context):
		context.scene.objects[self.objectName].pes_model_file = False
		return {'FINISHED'}

class PES_Model_Scene_Panel_Model_Export_Settings(bpy.types.Menu):
	"""Export Settings"""
	bl_label = "Export settings"
	
	def draw(self, context):
		self.layout.prop(context.active_object, 'pes_model_export_extensions_enabled')
		row = self.layout.row()
		row.prop(context.active_object, 'pes_model_export_mesh_names')
		row.enabled = context.active_object.pes_model_export_extensions_enabled
		row = self.layout.row()
		row.prop(context.active_object, 'pes_model_export_loop_preservation')
		row.enabled = context.active_object.pes_model_export_extensions_enabled
		row = self.layout.row()
		row.prop(context.active_object, 'pes_model_export_mesh_splitting')
		row.enabled = context.active_object.pes_model_export_extensions_enabled

class PES_Model_Scene_Panel_Model_Select_Filename(bpy.types.Operator):
	"""Select a filename to export this .model file"""
	bl_idname = "pes_model.exportable_select_filename"
	bl_label = "Select Filename"
	bl_options = {'UNDO', 'INTERNAL'}
	
	objectName = bpy.props.StringProperty(name = "Object to export")
	filepath = bpy.props.StringProperty(subtype = 'FILE_PATH')
	check_existing = bpy.props.BoolProperty(default = True)
	filter_glob = bpy.props.StringProperty(default = "*.model")
	
	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}
	
	def check(self, context):
		return True
	
	def execute(self, context):
		context.scene.objects[self.objectName].pes_model_filename = self.filepath
		return {'FINISHED'}

class PES_Model_Scene_Panel(bpy.types.Panel):
	bl_label = "PES .model I/O"
	bl_space_type = "PROPERTIES"
	bl_region_type = "WINDOW"
	bl_context = "scene"
	
	@classmethod
	def poll(cls, context):
		return context.scene != None
	
	def draw(self, context):
		scene = context.scene
		
		modelFileObjects = []
		for object in context.scene.objects:
			if object.pes_model_file:
				modelFileObjects.append(object)
		modelFileObjects.sort(key = lambda object: object.name)
		
		mainColumn = self.layout.column()
		importRow = mainColumn.row()
		buttonColumn = importRow.column()
		buttonColumn.operator(PES_Model_Scene_Import.bl_idname)
		importRow.menu(PES_Model_Scene_Panel_Model_Import_Settings.__name__, icon = 'DOWNARROW_HLT', text = "")
		
		for object in modelFileObjects:
			box = mainColumn.box()
			column = box.column()
			
			row = column.row()
			row.label("Object: %s" % object.name)
			#row.operator(FMDL_Scene_Panel_FMDL_Remove.bl_idname, text = "", icon = 'X').objectName = object.name
			
			row = column.row(align = True)
			row.prop(object, 'pes_model_filename', text = "Export Path")
			row.operator(PES_Model_Scene_Panel_Model_Select_Filename.bl_idname, text = "", icon = 'FILESEL').objectName = object.name
			
			row = column.row()
			row.operator_context = 'EXEC_DEFAULT'
			row.context_pointer_set('active_object', object)
			subrow = row.row()
			exportSettings = subrow.operator(PES_Model_Scene_Export_Object.bl_idname)
			exportSettings.objectName = object.name
			exportSettings.filepath = object.pes_model_filename
			exportSettings.extensions_enabled = object.pes_model_export_extensions_enabled
			exportSettings.mesh_names = object.pes_model_export_mesh_names
			exportSettings.loop_preservation = object.pes_model_export_loop_preservation
			exportSettings.mesh_splitting = object.pes_model_export_mesh_splitting
			if object.pes_model_filename == "":
				subrow.enabled = False
			#row.operator(PES_Model_Scene_Export_Object_Summary.bl_idname, text = "", icon = 'INFO').objectName = object.name
			row.menu(PES_Model_Scene_Panel_Model_Export_Settings.__name__, icon = 'DOWNARROW_HLT', text = "")

class PES_Model_Mesh_Panel(bpy.types.Panel):
	bl_label = "PES .model Mesh Settings"
	bl_space_type = "PROPERTIES"
	bl_region_type = "WINDOW"
	bl_context = "data"
	
	@classmethod
	def poll(cls, context):
		return context.mesh != None
	
	def draw(self, context):
		mesh = context.mesh
		
		mainColumn = self.layout.column()
		mainColumn.prop(mesh, "pes_model_material")



def PES_Model_Armature_SkeletonType_get(armatureObject):
	return 0 if armatureObject.data.pes_model_simplified_skeleton else 1

def PES_Model_Armature_SkeletonType_set(armatureObject, value):
	if value == 1:
		if armatureObject.data.pes_model_simplified_skeleton:
			Skeleton.convertSimplifiedSkeletonToReal(bpy.context, armatureObject)
		armatureObject.data.pes_model_simplified_skeleton = False
	else:
		if not armatureObject.data.pes_model_simplified_skeleton:
			Skeleton.convertRealSkeletonToSimplified(bpy.context, armatureObject)
		armatureObject.data.pes_model_simplified_skeleton = True

class PES_Model_Armature_StandardizePose(bpy.types.Operator):
	"""Pose skeleton to standard PES T-pose"""
	bl_idname = "pes_model.standardize_pose"
	bl_label = "Pose skeleton to standard PES pose"
	bl_options = {'REGISTER', 'UNDO'}
	
	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT' and context.active_object is not None and context.active_object.type == 'ARMATURE'
	
	def execute(self, context):
		if context.active_object.data.pes_model_simplified_skeleton:
			Skeleton.reposeSimplifiedSkeletonToStandard(context, context.active_object)
		else:
			armatureObjectName = context.active_object.name
			Skeleton.convertRealSkeletonToSimplified(context, bpy.data.objects[armatureObjectName], convertPose = False)
			Skeleton.reposeSimplifiedSkeletonToStandard(context, bpy.data.objects[armatureObjectName])
			Skeleton.convertSimplifiedSkeletonToReal(context, bpy.data.objects[armatureObjectName], convertPose = False)
		return {'FINISHED'}

class PES_Model_Armature_Panel(bpy.types.Panel):
	bl_label = "PES .model Skeleton Settings"
	bl_space_type = "PROPERTIES"
	bl_region_type = "WINDOW"
	bl_context = "data"
	
	@classmethod
	def poll(cls, context):
		return context.armature != None
	
	def draw(self, context):
		armature = context.armature
		
		mainColumn = self.layout.column()
		mainColumn.prop(context.object, 'pes_model_skeleton_type')
		mainColumn.operator(PES_Model_Armature_StandardizePose.bl_idname, icon = 'ARMATURE_DATA')



classes = [
	PES_Model_Scene_Import,
	PES_Model_Scene_Export_Object,
	PES_Model_Scene_Panel_Model_Import_Settings,
	PES_Model_Scene_Panel_Model_Remove,
	PES_Model_Scene_Panel_Model_Export_Settings,
	PES_Model_Scene_Panel_Model_Select_Filename,
	PES_Model_Scene_Panel,
	PES_Model_Mesh_Panel,
	PES_Model_Armature_StandardizePose,
	PES_Model_Armature_Panel,
]



def register():
	bpy.types.Object.pes_model_file = bpy.props.BoolProperty(name = "Is .model file", options = {'SKIP_SAVE'})
	bpy.types.Object.pes_model_filename = bpy.props.StringProperty(name = ".model filename", options = {'SKIP_SAVE'})
	bpy.types.Object.pes_model_export_extensions_enabled = bpy.props.BoolProperty(name = "Enable blender-pes-model extensions", default = True)
	bpy.types.Object.pes_model_export_mesh_names = bpy.props.BoolProperty(name = "Store mesh names", default = True)
	bpy.types.Object.pes_model_export_loop_preservation = bpy.props.BoolProperty(name = "Preserve split vertices", default = True)
	bpy.types.Object.pes_model_export_mesh_splitting = bpy.props.BoolProperty(name = "Autosplit overlarge meshes", default = True)
	bpy.types.Scene.pes_model_import_extensions_enabled = bpy.props.BoolProperty(name = "Enable blender-pes-model extensions", default = True)
	bpy.types.Scene.pes_model_import_mesh_names = bpy.props.BoolProperty(name = "Store mesh names", default = True)
	bpy.types.Scene.pes_model_import_loop_preservation = bpy.props.BoolProperty(name = "Preserve split vertices", default = True)
	bpy.types.Scene.pes_model_import_mesh_splitting = bpy.props.BoolProperty(name = "Autosplit overlarge meshes", default = True)
	bpy.types.Scene.pes_model_import_skeleton_simplification = bpy.props.BoolProperty(name = "Import simplified skeletons if possible", default = True)
	bpy.types.Scene.pes_model_ignore_parser_warnings = bpy.props.BoolProperty(name = "Ignore parser warnings", default = False)
	bpy.types.Scene.pes_model_lenient_parsing = bpy.props.BoolProperty(name = "Fix corruptions where possible", default = True)
	
	bpy.types.Object.pes_model_skeleton_type = bpy.props.EnumProperty(name = "Skeleton mode",
		items = [
			('simplified', 'Simplified', 'Simplified'),
			('real', 'Real', 'Real'),
		],
		get = PES_Model_Armature_SkeletonType_get,
		set = PES_Model_Armature_SkeletonType_set,
		options = {'SKIP_SAVE'}
	)
	
	for c in classes:
		bpy.utils.register_class(c)
	
	bpy.types.INFO_MT_file_import.append(PES_Model_Scene_Import_MenuItem)

def unregister():
	bpy.types.INFO_MT_file_import.remove(PES_Model_Scene_Import_MenuItem)
	
	for c in classes[::-1]:
		bpy.utils.unregister_class(c)
