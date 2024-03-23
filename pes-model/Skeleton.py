import bpy
import math
import mathutils
import numpy
from . import PesSkeletonData

def pesToNumpy(matrix):
	return numpy.array([
		matrix[0:4],
		matrix[4:8],
		matrix[8:12],
		[0, 0, 0, 1],
	])

def blenderToNumpy(matrix):
	return numpy.array([[row[i] for i in range(4)] for row in matrix])

def numpyToBlender(matrix):
	return mathutils.Matrix([[matrix[i][j] for j in range(4)] for i in range(4)])

def matmul(*args):
	accumulator = args[0]
	for arg in args[1:]:
		accumulator = numpy.matmul(accumulator, arg)
	return accumulator

#
# In PES:
# - the x coordinate runs from right (negative) to left (positive);
# - the y coordinate runs from bottom (negative) to top (positive);
# - the z coordinate runs from back (negative) to front (positive).
# In blender:
# - the x coordinate runs from right (negative) to left (positive);
# - the y coordinate runs from front (negative) to back (positive);
# - the z coordinate runs from bottom (negative) to top (positive).
#

pesToBlenderTransformation = numpy.array([
	[1, 0, 0, 0],
	[0, 0, -1, 0],
	[0, 1, 0, 0],
	[0, 0, 0, 1],
])

#
# In PES bone matrices:
# - the x column is the bone vector;
# - the y column is the binormal;
# - the z column is the normal;
# - the w column is the bone starting position.
# In blender bone matrices:
# - the x column is the normal;
# - the y column is the bone vector;
# - the z column is the binormal;
# - the w column is the bone starting position.
#

pesToBlenderBoneMatrixTransformation = numpy.array([
	[0, 0, 1, 0],
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 0, 1],
])

def boneMatrixPesToBlender(pesBoneMatrix):
	return matmul(
		pesToBlenderTransformation,
		pesBoneMatrix,
		numpy.linalg.inv(pesToBlenderTransformation),
		numpy.linalg.inv(pesToBlenderBoneMatrixTransformation),
	)

def boneMatrixBlenderToPes(blenderBoneMatrix):
	return matmul(
		numpy.linalg.inv(pesToBlenderTransformation),
		blenderBoneMatrix,
		pesToBlenderBoneMatrixTransformation,
		pesToBlenderTransformation,
	)



def makeDesiredSimplifiedBlenderBoneMatrix(armature, databaseBone):
	editBone = armature.edit_bones.new("temp")
	(headX, headY, headZ) = databaseBone.startPosition
	(tailX, tailY, tailZ) = databaseBone.endPosition
	editBone.head = (headX, -headZ, headY)
	editBone.tail = (tailX, -tailZ, tailY)
	editBone.roll = 0
	matrix = editBone.matrix
	armature.edit_bones.remove(editBone)
	return blenderToNumpy(matrix)

_simplifiedDatabaseBoneMatrices = None

def computeSimplifiedDatabaseBoneMatrices(context):
	global _simplifiedDatabaseBoneMatrices
	if _simplifiedDatabaseBoneMatrices is not None:
		return
	
	if context.scene.objects.active is None:
		activeObjectName = None
	else:
		activeObjectName = context.scene.objects.active.name
	
	armature = bpy.data.armatures.new("tempComputation")
	armatureName = armature.name
	armatureObject = bpy.data.objects.new("tempComputation", armature)
	armatureObjectName = armatureObject.name
	context.scene.objects.link(armatureObject)
	context.scene.objects.active = armatureObject
	
	c = context.copy()
	c["object"] = armatureObject
	c["active_object"] = armatureObject
	bpy.ops.object.mode_set(c, mode = 'EDIT')
	c = None
	
	_simplifiedDatabaseBoneMatrices = {}
	for databaseBone in PesSkeletonData.bones.values():
		_simplifiedDatabaseBoneMatrices[databaseBone.name] = makeDesiredSimplifiedBlenderBoneMatrix(armature, databaseBone)
	
	c = context.copy()
	c["object"] = armatureObject
	c["active_object"] = armatureObject
	bpy.ops.object.mode_set(c, mode = 'OBJECT')
	c = None
	
	if activeObjectName is None:
		context.scene.objects.active = None
	else:
		context.scene.objects.active = bpy.data.objects[activeObjectName]
	
	armatureObject = None
	armature = None
	context.scene.objects.unlink(bpy.data.objects[armatureObjectName])
	bpy.data.objects.remove(bpy.data.objects[armatureObjectName])
	bpy.data.armatures.remove(bpy.data.armatures[armatureName])

def simplifiedDatabaseBoneMatrix(name):
	global _simplifiedDatabaseBoneMatrices
	return _simplifiedDatabaseBoneMatrices[name]

def boneMatrixRealToSimplified(databaseBone, realBoneMatrix):
	return matmul(
		realBoneMatrix,
		numpy.linalg.inv(boneMatrixPesToBlender(pesToNumpy(databaseBone.matrix))),
		simplifiedDatabaseBoneMatrix(databaseBone.name),
	)

def boneMatrixSimplifiedToReal(databaseBone, simplifiedBoneMatrix):
	return matmul(
		simplifiedBoneMatrix,
		numpy.linalg.inv(simplifiedDatabaseBoneMatrix(databaseBone.name)),
		boneMatrixPesToBlender(pesToNumpy(databaseBone.matrix)),
	)



def databaseBoneLength(databaseBone):
	return sum((databaseBone.startPosition[i] - databaseBone.endPosition[i]) ** 2 for i in range(3)) ** 0.5

def setBoneGeometry(bone, matrix, length):
	bone.head = tuple(matrix[i][3] for i in range(3))
	bone.tail = tuple(matrix[i][3] + matrix[i][1] * length for i in range(3))
	
	#
	# To set the normal and binormal, we need to modify the bone's roll, i.e. the angle of rotation along the
	# bone vector. It should be set so that the blender bone normal lines up with the specified matrix's normal.
	#
	# I have no idea what the roll is relative to, i.e. what roll=0 specifies. So we set roll=0 first, compute
	# the angle between the resulting blender's bone normal and the desired one, then set roll=-angle.
	#
	bone.roll = 0
	
	#
	# To compute the angle between the desired and found normals, first we compute the cosine using the dot
	# product, which determines the angle up to sign. To find the sign, we compute the cross product, and
	# dot product that against the bone vector; this is the sine of the angle. atan2() then gives the exact angle.
	#
	axis = tuple(matrix[i][1] for i in range(3))
	desiredNormal = tuple(matrix[i][0] for i in range(3))
	currentNormal = tuple(bone.matrix[i][0] for i in range(3))
	
	normalDotProduct = sum(desiredNormal[i] * currentNormal[i] for i in range(3))
	normalCrossProduct = (
		desiredNormal[2] * currentNormal[1] - desiredNormal[1] * currentNormal[2],
		desiredNormal[0] * currentNormal[2] - desiredNormal[2] * currentNormal[0],
		desiredNormal[1] * currentNormal[0] - desiredNormal[0] * currentNormal[1],
	)
	sine = sum(axis[i] * normalCrossProduct[i] for i in range(3))
	angle = math.atan2(sine, normalDotProduct)
	
	bone.roll = angle

def disconnectBoneParents(armature):
	for name in [bone.name for bone in armature.edit_bones]:
		armature.edit_bones[name].use_connect = False

def connectBoneParents(armature):
	for name in [bone.name for bone in armature.edit_bones]:
		if armature.edit_bones[name].parent is not None:
			parentDistanceSquared = sum((armature.edit_bones[name].head[i] - armature.edit_bones[name].parent.tail[i]) ** 2 for i in range(3))
			if parentDistanceSquared < 0.0000000001:
				armature.edit_bones[name].use_connect = True

def setBonePosesToMatrices(pose, desiredPoseMatrices):
	posedBones = set()
	def poseBone(name):
		if name in posedBones:
			return
		
		if pose.bones[name].parent is not None:
			poseBone(pose.bones[name].parent.name)
		
		if name in desiredPoseMatrices:
			currentPoseMatrix = blenderToNumpy(pose.bones[name].matrix)
			currentRelativePoseMatrix = blenderToNumpy(pose.bones[name].matrix_basis)
			currentPosedBoneMatrix = matmul(
				currentPoseMatrix,
				numpy.linalg.inv(currentRelativePoseMatrix)
			)
			desiredRelativePoseMatrix = matmul(
				numpy.linalg.inv(currentPosedBoneMatrix),
				desiredPoseMatrices[name],
			)
			pose.bones[name].matrix_basis = numpyToBlender(desiredRelativePoseMatrix)
		
		# Blender doesn't update descendent bone matrices without this.
		bpy.context.scene.update()
		posedBones.add(name)
	
	for name in desiredPoseMatrices.keys():
		poseBone(name)



def convertRealSkeletonToSimplified(context, armatureObject, convertPose = True, connectChildren = False):
	# Compute in object mode, for the pose is invisible in edit mode
	currentPoseMatrices = { bone.name : blenderToNumpy(bone.matrix) for bone in armatureObject.pose.bones }
	
	computeSimplifiedDatabaseBoneMatrices(context)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'EDIT')
	
	disconnectBoneParents(armatureObject.data)
	
	desiredPoseMatrices = {}
	for name in [bone.name for bone in armatureObject.data.edit_bones]:
		if name not in PesSkeletonData.bones:
			print("Not converting unknown bone '%s'" % name)
			continue
		
		databaseBone = PesSkeletonData.bones[name]
		currentBoneMatrix = blenderToNumpy(armatureObject.data.edit_bones[name].matrix)
		desiredBoneMatrix = boneMatrixRealToSimplified(databaseBone, currentBoneMatrix)
		desiredPoseMatrices[name] = boneMatrixRealToSimplified(databaseBone, currentPoseMatrices[name])
		
		setBoneGeometry(armatureObject.data.edit_bones[name], desiredBoneMatrix, databaseBoneLength(databaseBone))
	
	if connectChildren:
		connectBoneParents(armatureObject.data)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	
	if convertPose:
		setBonePosesToMatrices(armatureObject.pose, desiredPoseMatrices)

def convertSimplifiedSkeletonToReal(context, armatureObject, convertPose = True):
	# Compute in object mode, for the pose is invisible in edit mode
	currentPoseMatrices = { bone.name : blenderToNumpy(bone.matrix) for bone in armatureObject.pose.bones }
	
	computeSimplifiedDatabaseBoneMatrices(context)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'EDIT')
	
	disconnectBoneParents(armatureObject.data)
	
	desiredPoseMatrices = {}
	for name in [bone.name for bone in armatureObject.data.edit_bones]:
		if name not in PesSkeletonData.bones:
			print("Not converting unknown bone '%s'" % name)
			continue
		
		databaseBone = PesSkeletonData.bones[name]
		currentBoneMatrix = blenderToNumpy(armatureObject.data.edit_bones[name].matrix)
		desiredBoneMatrix = boneMatrixSimplifiedToReal(databaseBone, currentBoneMatrix)
		desiredPoseMatrices[name] = boneMatrixSimplifiedToReal(databaseBone, currentPoseMatrices[name])
		
		setBoneGeometry(armatureObject.data.edit_bones[name], desiredBoneMatrix, databaseBoneLength(databaseBone) * 0.8)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	
	if convertPose:
		setBonePosesToMatrices(armatureObject.pose, desiredPoseMatrices)

def reposeSimplifiedSkeletonToStandard(context, armatureObject):
	computeSimplifiedDatabaseBoneMatrices(context)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'EDIT')
	
	disconnectBoneParents(armatureObject.data)
	
	# This is required. Going directly from edit mode to pose mode causes a segfault.
	bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
	
	desiredBoneMatrices = {}
	for name in [bone.name for bone in armatureObject.data.bones]:
		if name in PesSkeletonData.bones:
			desiredBoneMatrices[name] = simplifiedDatabaseBoneMatrix(name)
		else:
			desiredBoneMatrices[name] = numpy.array([
				[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1],
			])
	setBonePosesToMatrices(armatureObject.pose, desiredBoneMatrices)
	
	#
	# Apply the current pose to all influenced meshes as well as the armature itself.
	#
	
	for blenderObject in context.scene.objects:
		if blenderObject.type != 'MESH':
			continue
		
		modifierName = None
		modifierIndex = None
		for i in range(len(blenderObject.modifiers)):
			modifier = blenderObject.modifiers[i]
			if modifier.type == 'ARMATURE' and modifier.object.name == armatureObject.name:
				modifierName = modifier.name
				modifierIndex = i
				break
		if modifierName is None:
			continue
		
		c = context.copy()
		c['object'] = blenderObject
		c['modifier'] = blenderObject.modifiers[modifierName]
		bpy.ops.object.modifier_copy(c, modifier = modifierName)
		
		c = context.copy()
		c['object'] = blenderObject
		c['modifier'] = blenderObject.modifiers[modifierName]
		bpy.ops.object.modifier_move_down(c, modifier = modifierName)
		
		c = context.copy()
		c['object'] = blenderObject
		c['modifier'] = blenderObject.modifiers[modifierIndex]
		bpy.ops.object.modifier_apply(c, modifier = blenderObject.modifiers[modifierIndex].name)
	
	bpy.ops.object.mode_set(context.copy(), mode = 'POSE')
	
	bpy.ops.pose.armature_apply(context.copy())
	
	bpy.ops.object.mode_set(context.copy(), mode = 'OBJECT')
