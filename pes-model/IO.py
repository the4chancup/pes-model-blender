import bpy
import math
import mathutils
import itertools
import os
import os.path
import re

from . import ModelFile

import numpy


class ExportError(Exception):
	def __init__(self, errors):
		if isinstance(errors, list):
			self.errors = errors
		else:
			self.errors = [errors]

def setActiveObject(context, blenderObject):
	if 'view_layer' in dir(context):
		context.view_layer.objects.active = blenderObject
	else:
		context.scene.objects.active = blenderObject



def importModel(context, model, filename):
	UV_MAP_COLOR = 'UVMap'
	UV_MAP_NORMALS = 'normal_map'
	
	def addBone(blenderArmature, bone, boneIDs, bonesByName):
		if bone in boneIDs:
			return boneIDs[bone]
		
		blenderEditBone = blenderArmature.edit_bones.new(bone.name)
		boneID = blenderEditBone.name
		boneIDs[bone] = boneID
		
		# TODO: Make two versions of this, one based on the bone matrix and one based on hardcoded skeleton data
		
		#
		# EXPERIMENT -- compute bone geometry from .model bone matrix
		#
		
		#
		# bone.matrix is the matrix that transforms coordinates from model space to bone space.
		# That is, it is the inverse of the matrix that blender uses to define the bone.
		#
		boneMatrix = numpy.array([
			bone.matrix[0:4],
			bone.matrix[4:8],
			bone.matrix[8:12],
			[0, 0, 0, 1],
		])
		inverse = numpy.linalg.inv(boneMatrix)
		
		#
		# Convert to blender's coordinate system
		#
		pesToBlenderTransformation = numpy.array([
			[1, 0, 0, 0],
			[0, 0, -1, 0],
			[0, 1, 0, 0],
			[0, 0, 0, 1],
		])
		effectiveBoneMatrix = numpy.matmul(pesToBlenderTransformation, numpy.matmul(inverse, numpy.linalg.inv(pesToBlenderTransformation)))
		
		#
		# We now have:
		# column 0: vector along the direction of the bone
		# column 1: binormal
		# column 2: normal
		# column 3: bone starting position
		#
		
		length = 0.2
		blenderEditBone.head = (effectiveBoneMatrix[0][3], effectiveBoneMatrix[1][3], effectiveBoneMatrix[2][3])
		blenderEditBone.tail = (
			effectiveBoneMatrix[0][3] + effectiveBoneMatrix[0][0] * length,
			effectiveBoneMatrix[1][3] + effectiveBoneMatrix[1][0] * length,
			effectiveBoneMatrix[2][3] + effectiveBoneMatrix[2][0] * length,
		)
		
		#
		# To set the normal and binormal, we need to modify the bone's roll, i.e. the angle of rotation along the
		# bone vector. It should be set so that the blender bone normal lines up with the specified matrix's normal.
		#
		# I have no idea what the roll is relative to, i.e. what roll=0 specifies. So we set roll=0 first, compute
		# the angle between the resulting blender's bone normal and the desired one, then set roll=-angle.
		#
		# The blender bone matrix has the bone vector in column 1 [y], and the normal in column 0 [x].
		#
		blenderEditBone.roll = 0
		normalDotProduct = sum(blenderEditBone.matrix[i][2] * effectiveBoneMatrix[i][1] for i in range(3))
		blenderEditBone.roll = math.acos(normalDotProduct)
		
		#
		# END EXPERIMENT
		#
		
		blenderEditBone.use_connect = False
		blenderEditBone.hide = False
		
		return boneID
	
	def importSkeleton(context, model):
		blenderArmature = bpy.data.armatures.new("Skeleton")
		blenderArmature.show_names = True
		
		blenderArmatureObject = bpy.data.objects.new("Skeleton", blenderArmature)
		armatureObjectID = blenderArmatureObject.name
		
		context.scene.objects.link(blenderArmatureObject)
		setActiveObject(context, blenderArmatureObject)
		
		bpy.ops.object.mode_set(context.copy(), mode = 'EDIT')
		
		bonesByName = {}
		for bone in model.bones:
			bonesByName[bone.name] = bone
		
		boneIDs = {}
		for bone in model.bones:
			addBone(blenderArmature, bone, boneIDs, bonesByName)
		
		bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
		
		return (armatureObjectID, boneIDs)
	
	def addSkeletonMeshModifier(blenderMeshObject, boneGroup, armatureObjectID, boneIDs):
		blenderArmatureObject = bpy.data.objects[armatureObjectID]
		blenderArmature = blenderArmatureObject.data
		
		blenderModifier = blenderMeshObject.modifiers.new("model skeleton", type = 'ARMATURE')
		blenderModifier.object = blenderArmatureObject
		blenderModifier.use_vertex_groups = True
		
		groupBoneIDs = []
		for bone in boneGroup.bones:
			boneID = boneIDs[bone]
			blenderBone = blenderArmature.bones[boneID]
			blenderVertexGroup = blenderMeshObject.vertex_groups.new(blenderBone.name)
			groupBoneIDs.append(blenderVertexGroup.name)
		return groupBoneIDs
	
	def importMesh(mesh, name, armatureObjectID, boneIDs):
		blenderMesh = bpy.data.meshes.new(name)
		
		#
		# mesh.vertices does not correspond either to the blenderMesh.vertices
		# nor the blenderMesh.loops, but rather the unique values of blenderMesh.loops.
		# The blenderMesh.vertices correspond to the unique vertex.position values in mesh.vertices.
		#
		
		vertexIndices = {}
		vertexVertices = []
		for vertex in mesh.vertices:
			if vertex.position not in vertexIndices:
				vertexIndices[vertex.position] = len(vertexIndices)
				vertexVertices.append(vertex)
		loopVertices = list(itertools.chain.from_iterable([face.vertices for face in mesh.faces]))
		
		blenderMesh.vertices.add(len(vertexVertices))
		blenderMesh.vertices.foreach_set("co", tuple(itertools.chain.from_iterable([
			(vertex.position.x, -vertex.position.z, vertex.position.y) for vertex in vertexVertices
		])))
		
		blenderMesh.loops.add(len(mesh.faces) * 3)
		blenderMesh.loops.foreach_set("vertex_index", tuple([vertexIndices[vertex.position] for vertex in loopVertices]))
		
		blenderMesh.polygons.add(len(mesh.faces))
		blenderMesh.polygons.foreach_set("loop_start", tuple(range(0, 3 * len(mesh.faces), 3)))
		blenderMesh.polygons.foreach_set("loop_total", [3 for face in mesh.faces])
		
		blenderMesh.update(calc_edges = True)
		
		if mesh.vertexFields.hasNormal:
			def normalize(vector):
				(x, y, z) = vector
				size = (x ** 2 + y ** 2 + z ** 2) ** 0.5
				if size < 0.01:
					return (x, y, z)
				return (x / size, y / size, z / size)
			blenderMesh.normals_split_custom_set([
				normalize((vertex.normal.x, -vertex.normal.z, vertex.normal.y)) for vertex in loopVertices
			])
			blenderMesh.use_auto_smooth = True
		
		if mesh.vertexFields.hasColor:
			colorLayer = blenderMesh.vertex_colors.new()
			colorLayer.data.foreach_set("color", tuple(itertools.chain.from_iterable([
				vertex.color[0:3] for vertex in loopVertices
			])))
			colorLayer.active = True
			colorLayer.active_render = True
		
		if mesh.vertexFields.uvCount >= 1:
			if 'uv_textures' in dir(blenderMesh):
				uvTexture = blenderMesh.uv_textures.new(name = UV_MAP_COLOR)
				uvLayer = blenderMesh.uv_layers[uvTexture.name]
			else:
				uvLayer = blenderMesh.uv_layers.new(name = UV_MAP_COLOR, do_init = False)
				uvTexture = uvLayer
			
			uvLayer.data.foreach_set("uv", tuple(itertools.chain.from_iterable([
				(vertex.uv[0].u, 1.0 - vertex.uv[0].v) for vertex in loopVertices
			])))
			uvTexture.active = True
			uvTexture.active_clone = True
			uvTexture.active_render = True
		
		#
		# TODO: uv equalities
		#
		if mesh.vertexFields.uvCount >= 2:
			if 'uv_textures' in dir(blenderMesh):
				uvTexture = blenderMesh.uv_textures.new(name = UV_MAP_NORMALS)
				uvLayer = blenderMesh.uv_layers[uvTexture.name]
			else:
				uvLayer = blenderMesh.uv_layers.new(name = UV_MAP_NORMALS, do_init = False)
				uvTexture = uvLayer
			
			uvLayer.data.foreach_set("uv", tuple(itertools.chain.from_iterable([
				(vertex.uv[1].u, 1.0 - vertex.uv[1].v) for vertex in loopVertices
			])))
		
		blenderMesh.pes_model_material = mesh.material
		
		blenderMeshObject = bpy.data.objects.new(blenderMesh.name, blenderMesh)
		meshObjectID = blenderMeshObject.name
		context.scene.objects.link(blenderMeshObject)
		
		if mesh.vertexFields.hasBoneMapping:
			groupBoneIDs = addSkeletonMeshModifier(blenderMeshObject, mesh.boneGroup, armatureObjectID, boneIDs)
			for i in range(len(vertexVertices)):
				for (boneIndex, weight) in vertexVertices[i].boneMapping.items():
					if boneIndex < len(groupBoneIDs):
						blenderMeshObject.vertex_groups[groupBoneIDs[boneIndex]].add((i, ), weight, 'REPLACE')
		
		return meshObjectID
	
	def importMeshes(context, model, armatureObjectID, boneIDs):
		meshObjectIDs = []
		meshIndex = 0
		for mesh in model.meshes:
			if mesh.name is not None:
				name = mesh.name
			else:
				while True:
					name = f"mesh {meshIndex}"
					meshIndex += 1
					if name not in bpy.data.objects and name not in bpy.data.meshes:
						break
				
			meshObjectIDs.append(importMesh(mesh, name, armatureObjectID, boneIDs))
		
		return meshObjectIDs
	
	def importMeshTree(context, meshObjectIDs, armatureObjectID, filename):
		dirname = os.path.basename(os.path.dirname(filename))
		basename = os.path.basename(filename)
		position = basename.rfind('.')
		if position == -1:
			name = os.path.join(dirname, basename)
		else:
			name = os.path.join(dirname, basename[:position])
		
		blenderMeshGroupObject = bpy.data.objects.new(name, None)
		blenderMeshGroupID = blenderMeshGroupObject.name
		context.scene.objects.link(blenderMeshGroupObject)
		for meshObjectID in meshObjectIDs:
			bpy.data.objects[meshObjectID].parent = blenderMeshGroupObject
		
		if armatureObjectID is not None:
			bpy.data.objects[armatureObjectID].parent = blenderMeshGroupObject
		
		bpy.data.objects[blenderMeshGroupID].pes_model_file = True
		bpy.data.objects[blenderMeshGroupID].pes_model_filename = filename
	
	
	
	if context.active_object == None:
		activeObjectID = None
	else:
		activeObjectID = bpy.data.objects.find(context.active_object.name)
	if context.mode != 'OBJECT':
		bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	
	
	
	if len(model.bones) > 0:
		(armatureObjectID, boneIDs) = importSkeleton(context, model)
	else:
		(armatureObjectID, boneIDs) = (None, [])
	
	meshObjectIDs = importMeshes(context, model, armatureObjectID, boneIDs)
	
	importMeshTree(context, meshObjectIDs, armatureObjectID, filename)
	
	
	
	if context.mode != 'OBJECT':
		bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	if activeObjectID != None:
		setActiveObject(context, bpy.data.objects[activeObjectID])

def exportModel(context, rootObjectName):
	def exportMaterials(blenderMeshObjects):
		materials = []
		for blenderMeshObject in blenderMeshObjects:
			material = blenderMeshObject.data.pes_model_material
			if material not in materials:
				materials.append(material)
		return materials
	
	def exportHardcodedBone(boneName):
		pass
	
	def exportBlenderBone(bone):
		blenderBoneMatrix = bone.matrix_local
		#
		# Transform from blender format:
		# - column x: normal
		# - column y: bone vector
		# - column z: binormal
		# - column w: head
		# To pes format:
		# - column x: bone vector
		# - column y: binormal
		# - column z: normal
		# - column w: head
		#
		blenderBoneMatrixPesFormat = numpy.array([[row[1], row[2], row[0], row[3]] for row in blenderBoneMatrix])
		
		#
		# To transform bone.matrix into the pes coordinate system, switch columns 2 and 3, and negate both;
		# then do the same with rows 2 and 3.
		#
		pesToBlenderTransformation = numpy.array([
			[1, 0, 0, 0],
			[0, 0, -1, 0],
			[0, 1, 0, 0],
			[0, 0, 0, 1],
		])
		pesBoneMatrix = numpy.matmul(numpy.linalg.inv(pesToBlenderTransformation), numpy.matmul(blenderBoneMatrixPesFormat, pesToBlenderTransformation))
		
		inverseBoneMatrix = numpy.linalg.inv(pesBoneMatrix)
		
		return ModelFile.ModelFile.Bone(bone.name, [v for row in inverseBoneMatrix for v in row][0:12])
	
	def exportUnknownBone(boneName):
		raise ExportError(f'Found vertex group {boneName} without corresponding bone')
	
	def findBone(boneName, candidateArmatures):
		for armature in candidateArmatures:
			for blenderBone in armature.bones:
				if blenderBone.name == boneName:
					return blenderBone
		return None
	
	def isHardcodedBone(boneName):
		return False
		# TODO: custom bone export setting
		#return boneName in PesSkeletonData.bones
	
	def exportBones(blenderMeshObjects, blenderArmatureObjects):
		blenderMeshLinkedArmatures = {}
		blenderLinkedArmatures = []
		for blenderMeshObject in blenderMeshObjects:
			blenderMeshLinkedArmatures[blenderMeshObject] = []
			for modifier in blenderMeshObject.modifiers:
				if modifier.type == 'ARMATURE':
					blenderArmature = modifier.object.data
					blenderLinkedArmatures.append(blenderArmature)
					blenderMeshLinkedArmatures[blenderMeshObject].append(blenderArmature)
		
		bones = []
		boneIndices = {}
		for blenderMeshObject in blenderMeshObjects:
			boneNames = [vertexGroup.name for vertexGroup in blenderMeshObject.vertex_groups]
			candidateArmatures = (
				blenderMeshLinkedArmatures[blenderMeshObject] +
				blenderLinkedArmatures +
				[ armatureObject.data for armatureObject in blenderArmatureObjects ]
			)
			for boneName in boneNames:
				if boneName not in boneIndices:
					if isHardcodedBone(boneName):
						bone = exportHardcodedBone(boneName)
					else:
						blenderBone = findBone(boneName, candidateArmatures)
						if blenderBone is not None:
							bone = exportBlenderBone(blenderBone)
						else:
							bone = exportUnknownBone(boneName)
					
					boneIndices[boneName] = len(bones)
					bones.append(bone)
		
		return (bones, boneIndices)
	
	def exportMeshGeometry(blenderMeshObject, colorLayer, uvLayerColor, uvLayerNormal, boneCount, scene):
		#
		# Setup a modified version of the mesh data that can be fiddled with.
		#
		modifiedBlenderMesh = blenderMeshObject.data.copy()
		
		#
		# Apply mesh-object position and orientation
		#
		modifiedBlenderMesh.transform(blenderMeshObject.matrix_world)
		
		loopTotals = [0 for i in range(len(modifiedBlenderMesh.polygons))]
		modifiedBlenderMesh.polygons.foreach_get("loop_total", loopTotals)
		if max(loopTotals) != 3:
			#
			# calc_tangents() only works on triangulated meshes
			#
			
			modifiedBlenderObject = bpy.data.objects.new('triangulation', modifiedBlenderMesh)
			modifiedBlenderObject.modifiers.new('triangulation', 'TRIANGULATE')
			newBlenderMesh = modifiedBlenderObject.to_mesh(scene, True, 'PREVIEW', calc_undeformed = True)
			bpy.data.objects.remove(modifiedBlenderObject)
			bpy.data.meshes.remove(modifiedBlenderMesh)
			modifiedBlenderMesh = newBlenderMesh
		
		modifiedBlenderMesh.use_auto_smooth = True
		if uvLayerNormal is None:
			uvLayerTangent = uvLayerColor
		else:
			uvLayerTangent = uvLayerNormal
		modifiedBlenderMesh.calc_tangents(uvLayerTangent)
		
		
		
		class Vertex:
			def __init__(self):
				self.position = None
				self.boneMapping = {}
				self.loops = []
		
		class Loop:
			def __init__(self):
				self.normal = None
				self.color = None
				self.uv = []
				
				self.tangents = []
				self.loopIndices = []
			
			def matches(self, other):
				if (self.color != None) != (other.color != None):
					return False
				if self.color != None and tuple(self.color) != tuple(other.color):
					return False
				if len(self.uv) != len(other.uv):
					return False
				for i in range(len(self.uv)):
					if self.uv[i].u != other.uv[i].u:
						return False
					if self.uv[i].v != other.uv[i].v:
						return False
				# Do an approximate check for normals.
				if self.normal.dot(other.normal) < 0.99:
					return False
				return True
			
			def add(self, other):
				self.tangents += other.tangents
				self.loopIndices += other.loopIndices
				self.normal = self.normal.slerp(other.normal, 1.0 / len(self.loopIndices))
			
			def computeTangent(self):
				# Filter out zero tangents
				nonzeroTangents = []
				for tangent in self.tangents:
					if tangent.length_squared > 0.1:
						nonzeroTangents.append(tangent)
				
				if len(nonzeroTangents) == 0:
					# Make up a tangent to avoid crashes
					# Cross product the loop normal with any vector that is not parallel with it.
					bestVector = None
					for v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
						vector = mathutils.Vector(v)
						if bestVector == None or abs(vector.dot(self.normal)) < abs(bestVector.dot(self.normal)):
							bestVector = vector
					return bestVector.cross(self.normal)
				
				# Average out the different tangents.
				# In case of conflicts, bias towards the first entry in the list.
				averageTangent = nonzeroTangents[0]
				weight = 1
				remaining = nonzeroTangents[1:]
				while len(remaining) > 0:
					skipped = []
					for tangent in remaining:
						if averageTangent.dot(tangent) < -0.9:
							skipped.append(tangent)
						else:
							weight += 1
							averageTangent = averageTangent.slerp(tangent, 1.0 / weight)
					if len(skipped) == len(remaining):
						break
					remaining = skipped
				return averageTangent
		
		#
		# For each vertex in the mesh, create a list of loops that refer to the vertex.
		# If multiple loops for the same vertex have the same vertex data, merge them.
		#
		# TODO: The tangent is treated as an irrelevant field here, with loops considered
		# identical even if their tangents are different. This is probably a mistake.
		#
		vertices = []
		for i in range(len(modifiedBlenderMesh.vertices)):
			blenderVertex = modifiedBlenderMesh.vertices[i]
			vertex = Vertex()
			vertex.position = ModelFile.ModelFile.Vector3(
				blenderVertex.co.x,
				blenderVertex.co.z,
				-blenderVertex.co.y,
			)
			for group in blenderVertex.groups:
				if group.group >= boneCount:
					continue
				vertex.boneMapping[group.group] = group.weight
			vertices.append(vertex)
		
		for i in range(len(modifiedBlenderMesh.loops)):
			blenderLoop = modifiedBlenderMesh.loops[i]
			vertex = vertices[blenderLoop.vertex_index]
			
			loop = Loop()
			loop.normal = blenderLoop.normal
			loop.tangents = [blenderLoop.tangent]
			loop.loopIndices = [i]
			
			if colorLayer is not None:
				loop.color = [c for c in modifiedBlenderMesh.vertex_colors[colorLayer].data[i].color] + [1.0]
			loop.uv.append(ModelFile.ModelFile.Vector2(
				modifiedBlenderMesh.uv_layers[uvLayerColor].data[i].uv[0],
				1.0 - modifiedBlenderMesh.uv_layers[uvLayerColor].data[i].uv[1],
			))
			if uvLayerNormal != None:
				loop.uv.append(ModelFile.ModelFile.Vector2(
					modifiedBlenderMesh.uv_layers[uvLayerNormal].data[i].uv[0],
					1.0 - modifiedBlenderMesh.uv_layers[uvLayerNormal].data[i].uv[1],
				))
			
			found = False
			for otherLoop in vertex.loops:
				if otherLoop.matches(loop):
					otherLoop.add(loop)
					found = True
					break
			if not found:
				vertex.loops.append(loop)
		
		#
		# For each loop in the mesh after coalescing duplicates, create one ModelFile.Vertex object.
		# Make sure that loops that belong to the same mesh vertex refer to the same ModelFile.Vertex.position object,
		# so that split-vertex-encoding can maintain this relation.
		#
		exportVertices = []
		exportLoopVertices = {}
		for vertex in vertices:
			for loop in vertex.loops:
				exportVertex = ModelFile.ModelFile.Vertex()
				exportVertex.position = vertex.position
				exportVertex.boneMapping = vertex.boneMapping
				exportVertex.normal = ModelFile.ModelFile.Vector3(
					loop.normal.x,
					loop.normal.z,
					-loop.normal.y,
				)
				tangent = loop.computeTangent()
				exportVertex.tangent = ModelFile.ModelFile.Vector3(
					tangent.x,
					tangent.z,
					-tangent.y,
				)
				exportVertex.color = loop.color
				exportVertex.uv = loop.uv
				exportVertices.append(exportVertex)
				for loopIndex in loop.loopIndices:
					exportLoopVertices[loopIndex] = exportVertex
		
		exportFaces = []
		for face in modifiedBlenderMesh.polygons:
			exportFaces.append(ModelFile.ModelFile.Face(
				exportLoopVertices[face.loop_start + 0],
				exportLoopVertices[face.loop_start + 1],
				exportLoopVertices[face.loop_start + 2],
			))
		
		bpy.data.meshes.remove(modifiedBlenderMesh)
		return (exportVertices, exportFaces)
	
	def exportMesh(blenderMeshObject, materials, bones, boneIndices, scene):
		blenderMesh = blenderMeshObject.data
		name = blenderMeshObject.name
		
		vertexFields = ModelFile.ModelFile.VertexFields()
		vertexFields.hasNormal = True
		vertexFields.hasTangent = True
		vertexFields.hasBitangent = False
		
		if len(blenderMesh.vertex_colors) == 0:
			colorLayer = None
			vertexFields.hasColor = False
		elif len(blenderMesh.vertex_colors) == 1:
			colorLayer = 0
			vertexFields.hasColor = True
		else:
			raise ExportError("Mesh '%s' has more than one color layer." % name)
		
		if len(blenderMesh.uv_layers) == 0:
			raise ExportError("Mesh '%s' does not have a UV map." % name)
		elif len(blenderMesh.uv_layers) == 1:
			uvLayerColor = blenderMesh.uv_layers[0].name
			uvLayerNormal = None
			vertexFields.uvCount = 1
		elif 'UVMap' in blenderMesh.uv_layers:
			uvLayerColor = 'UVMap'
			vertexFields.uvCount = 1
			if 'normal_map' in blenderMesh.uv_layers:
				uvLayerNormal = 'normal_map'
				vertexFields.uvCount += 1
			else:
				uvLayerNormal = None
		else:
			raise ExportError("Mesh '%s' has ambiguous UV maps: multiple UV maps found, none of which are called `UVMap`." % name)
		
		boneGroupBones = [bones[boneIndices[vertexGroup.name]] for vertexGroup in blenderMeshObject.vertex_groups]
		if len(boneGroupBones) > 0:
			vertexFields.hasBoneMapping = True
			boneGroup = ModelFile.ModelFile.BoneGroup()
			boneGroup.bones = boneGroupBones
		else:
			boneGroup = None
		
		(vertices, faces) = exportMeshGeometry(blenderMeshObject, colorLayer, uvLayerColor, uvLayerNormal, len(boneGroupBones), scene)
		
		if len(vertices) == 0:
			boundingBox = None
		else:
			boundingBox = ModelFile.ModelFile.BoundingBox(
				ModelFile.ModelFile.Vector3(
					min(vertex.position.x for vertex in vertices),
					min(vertex.position.y for vertex in vertices),
					min(vertex.position.z for vertex in vertices),
				),
				ModelFile.ModelFile.Vector3(
					max(vertex.position.x for vertex in vertices),
					max(vertex.position.y for vertex in vertices),
					max(vertex.position.z for vertex in vertices),
				),
			)
		
		mesh = ModelFile.ModelFile.Mesh()
		mesh.vertices = vertices
		mesh.faces = faces
		mesh.boneGroup = boneGroup
		mesh.material = blenderMesh.pes_model_material
		mesh.vertexFields = vertexFields
		mesh.boundingBox = boundingBox
		mesh.name = blenderMeshObject.name
		
		return mesh
	
	def calculateBoundingBox(meshes):
		boundingBoxes = [mesh.boundingBox for mesh in meshes if mesh.boundingBox is not None]
		
		if len(boundingBoxes) == 0:
			return ModelFile.ModelFile.BoundingBox(
				ModelFile.ModelFile.Vector3(0, 0, 0),
				ModelFile.ModelFile.Vector3(0, 0, 0),
			)
		
		return ModelFile.ModelFile.BoundingBox(
			ModelFile.ModelFile.Vector3(
				min(box.min.x for box in boundingBoxes),
				min(box.min.y for box in boundingBoxes),
				min(box.min.z for box in boundingBoxes),
			),
			ModelFile.ModelFile.Vector3(
				max(box.max.x for box in boundingBoxes),
				max(box.max.y for box in boundingBoxes),
				max(box.max.z for box in boundingBoxes),
			),
		)
	
	def listMeshObjects(context, rootObjectName):
		blenderMeshObjects = []
		blenderArmatureObjects = []
		def findMeshObjects(blenderObject, blenderMeshObjects):
			if blenderObject.type == 'MESH' and len(blenderObject.data.polygons) > 0:
				blenderMeshObjects.append(blenderObject)
			if blenderObject.type == 'ARMATURE':
				blenderArmatureObjects.append(blenderObject)
			childNames = [child.name for child in blenderObject.children]
			for childName in sorted(childNames):
				findMeshObjects(bpy.data.objects[childName], blenderMeshObjects)
		findMeshObjects(context.scene.objects[rootObjectName], blenderMeshObjects)
		
		return blenderMeshObjects, blenderArmatureObjects
	
	
	
	if context.mode != 'OBJECT':
		bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	
	
	
	(blenderMeshObjects, blenderArmatureObjects) = listMeshObjects(context, rootObjectName)
	
	materials = exportMaterials(blenderMeshObjects)
	
	(bones, boneIndices) = exportBones(blenderMeshObjects, blenderArmatureObjects)
	
	meshes = []
	for blenderMeshObject in blenderMeshObjects:
		meshes.append(exportMesh(blenderMeshObject, materials, bones, boneIndices, context.scene))
	
	boundingBox = calculateBoundingBox(meshes)
	
	model = ModelFile.ModelFile()
	model.bones = bones
	model.materials = materials
	model.meshes = meshes
	model.boundingBox = boundingBox
	
	errors = []
	for mesh in model.meshes:
		if len(mesh.vertices) > 65535:
			errors.append("Mesh '%s' contains %s vertices out of a maximum of 65535" % (meshName, len(mesh.vertices)))
		#if len(mesh.faces) > 21845:
		#	errors.append("Mesh '%s' contains %s faces out of a maximum of 21845" % (meshName, len(mesh.faces)))
		#if mesh.boneGroup is not None and len(mesh.boneGroup.bones) > 32:
		#	errors.append("Mesh '%s' bone group contains %s bones out of a maximum of 32" % (meshName, len(mesh.boneGroup.bones)))
	if len(errors) > 0:
		raise ExportError(errors)
	
	return model
