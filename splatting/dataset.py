from operator import invert, pos
import os
import struct
import math

import numpy as np
import torch as th
from torch._C import device
#from torchvision.transforms.functional import scale
import yaml
from PIL import Image, ImageOps
from plyfile import PlyData
from typing import Any, Tuple
from scipy.spatial.transform import Rotation as R
import array
import json
from .extensions.utils import normalize

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import speedups
speedups.enable()

@th.no_grad()
def createLightDirections(rings, points, mode, dtype, device):
    end = math.pi
    if mode == 'half':
        end = math.pi/2    
    # Create regular sampled angles
    x = np.linspace(0.0, end, num=rings, endpoint=True, retstep=False, dtype=None, axis=0)
    y = np.linspace(0.0, 2.0*math.pi, num=points, endpoint=True, retstep=False, dtype=None, axis=0)
    xx, yy = np.meshgrid(x,y)
    
    # Map angles to points on a sphere
    sphere = np.stack([
        np.sin(xx)*np.sin(yy),
        np.sin(xx)*np.cos(yy),
        np.cos(xx) ])
    sphere = np.moveaxis(sphere, 0, -1)
    sphere = np.reshape(sphere, [rings*points,3])
    sphere = sphere.astype(np.float32)
    return th.from_numpy(sphere).view(1,rings*points, 3).to(device, dtype)

def loadSceneLighting(filepath, dtype, device):
    # Reading from json file
    with open(os.path.join(filepath, 'lights.json'), 'r') as openfile:
        lighting = json.load(openfile)

    ambient = th.tensor(lighting['Ambient'], dtype=dtype, device=device).view(1,1,3)
    lights = lighting['Lights']
    if len(lights) > 0:
        directions = th.empty([1, len(lights), 3], dtype=dtype, device=device)
        colors = th.empty([1, len(lights), 3], dtype=dtype, device=device)
        for idx, light in enumerate(lights):
            directions[0,idx,:] = th.tensor(light['Direction'], dtype=dtype, device=device)
            colors[0,idx,:] = math.pi*th.tensor(light['Color'], dtype=dtype, device=device)
    else:
        directions = th.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).view(1,1,3)
        colors = th.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device).view(1,1,3)
    directions = normalize(directions)
    return ambient.contiguous(), directions.contiguous(), colors.contiguous()

def setupTrainablePointset(vertices : np.array, normals : np.array, colors : np.array, stdDevLimits : Tuple, dtype : Any, device : str) -> Tuple[th.nn.Parameter, th.nn.Parameter, th.nn.Parameter, th.nn.Parameter]:
    vertices = th.from_numpy(vertices).unsqueeze(0)
    vertices = th.nn.Parameter(vertices.to(device, dtype))
    normals = th.from_numpy(normals).unsqueeze(0)
    normals = th.nn.Parameter(normals.to(device, dtype))
    colors = th.from_numpy(colors).unsqueeze(0)
    colors = th.nn.Parameter(colors.to(device, dtype))

    l, r = stdDevLimits
    stdDevs = (l - r)*th.rand(size=[1, vertices.shape[1], 2], dtype=dtype) + r
    stdDevs = th.nn.Parameter(stdDevs.to(device, dtype))

    return vertices, normals, colors, stdDevs

def _readPoses(filepath : str, dtype : Any) -> np.array:
    matrices = list()
    with open(filepath, "rb") as f:
        for pose in iter(lambda: f.read(72), b''):
            frame = int.from_bytes(pose[:8], 'little')
            matrix = np.array(
                [struct.unpack('f', pose[8+4*i: 8+4*(i+1)]) for i in range(16)])
            matrix = np.reshape(matrix, [4, 4]).astype(dtype)
            matrices.append(matrix)
    return matrices

def _readFramePaths(path, framelist):
    filenames = list()
    with open(os.path.join(path, framelist), mode='r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            timestamp, filename = line.split(' ')
            filenames.append((float(timestamp), filename))
    filenames.sort(key=lambda x: x[0])
    frame_paths = [os.path.join(path, fn[:-1]) for _, fn in filenames]
    return frame_paths

def _loadPointset(filepath : str, dtype : Any) -> Tuple[np.array, np.array, np.array]:
    plydata = PlyData.read(filepath)
    vertices = np.stack([plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']], axis=-1).astype(dtype)
    normals = np.stack([plydata['vertex'].data['nx'], plydata['vertex'].data['ny'], plydata['vertex'].data['nz']], axis=-1).astype(dtype)
    try:
        colors = np.stack([plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue']], axis=-1).astype(dtype)
    except:
        colors = np.ones_like(normals)*255.0
    return vertices, normals, colors/255.0

def loadPointset(filepath : str, dtype : Any):
    return _loadPointset(filepath, dtype)

def _writePointsetAsPly(filename : str, points : np.array, normals : np.array, colors : np.array) -> None:
    header = "ply\n" \
        "format binary_little_endian 1.0\n" \
        "comment VCGLIB generated\n" \
        "element vertex {}\n" \
        "property float x\n" \
        "property float y\n" \
        "property float z\n" \
        "property float nx\n" \
        "property float ny\n" \
        "property float nz\n" \
        "property uchar red\n" \
        "property uchar green\n" \
        "property uchar blue\n" \
        "property uchar alpha\n" \
        "element face 0\n" \
        "property list uchar int vertex_indices\n" \
        "end_header\n".format(len(points))
    with open(filename, mode="wb") as file:
        file.write(header.encode('ascii'))
        for p, n, c in zip(points, normals, colors):
            array.array('f', p.tolist()).tofile(file)
            array.array('f', n.tolist()).tofile(file)
            for value in c:
                value = int(max(0.0, min(value, 1.0))*255)
                #value = min(int(value*255.0), 255)
                file.write(value.to_bytes(1, 'little'))
            alpha = 1
            file.write(alpha.to_bytes(1, 'little'))

def storePointset(filename : str, vertices : th.Tensor, normals : th.Tensor, colors : th.Tensor) -> None:
    vertices = vertices.detach().squeeze(0).numpy()
    normals = normals.detach().squeeze(0).numpy()
    colors = colors.detach().squeeze(0).numpy()
    _writePointsetAsPly(filename, vertices, normals, colors)

def _loadCameraMetadata(filepath : str) -> Tuple[int, int, float, float, float]:
    with open(filepath, mode='r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
        calibration = metadata['calibration_geometric']['intrinsics_color']
        focallengths = calibration['focal_lengths']
        assert focallengths['x'] == focallengths['y']
        focallength = focallengths['x']
        resolution = calibration['resolution']
        width = resolution['x']
        height = resolution['y']
        viewdistance = metadata['range']
        zmin = viewdistance['min']
        zmax = viewdistance['max']
    return width, height, focallength, zmin, zmax

def nearest_orthogonal(M):
    U, S, Vh = np.linalg.svd(M, full_matrices=True)
    return U @ Vh

class Dataset(th.utils.data.Dataset):
    def __init__(self, path : str, invertPose : bool, flipImage : bool, scale : int = -1, dtype = np.float32, pointset = "pointset.ply", verbose=True) -> None:
        self._root = path
        self._invert = invertPose
        self._flip = flipImage
        self._dtype = dtype
        self._pointset = pointset
        self._scale = scale

        # Load camera poses for all frames
        self._poses = _readPoses(os.path.join(path, "pose.dat"), np.float32)

        # Gather sorted rgb and depth frame file paths
        self._rgb_paths = _readFramePaths(path, "rgb.txt")
        self._depth_paths = _readFramePaths(path, "depth.txt")

        # Drops frames which are not present in the pose (assumes that frames and poses have the same order)
        numElements = min([len(self._rgb_paths), len(self._depth_paths), len(self._poses)])
        self._poses = self._poses[:numElements]
        self._rgb_paths = self._rgb_paths[:numElements]
        self._depth_paths = self._depth_paths[:numElements]
        
        if verbose is True:
            print("Dataset: found {} images".format(len(self._rgb_paths)))

    def _loadImage(self, filepath):
        image = Image.open(filepath, mode='r')
        if self._flip is True:
            image = ImageOps.flip(image)
        if self._scale > -1:
            image = image.resize((self._scale, self._scale), Image.BILINEAR)
        image = np.array(image, dtype=self._dtype)/255.0
        if len(image.shape) == 3:
            image = np.transpose(image, (1,0,2))
        else:
            image = np.transpose(image, (1,0))
        return image

    def getPointset(self) -> Tuple[np.array, np.array, np.array]:
        return _loadPointset(os.path.join(self._root, self._pointset), np.float32)

    def getCameraMetadata(self) -> Tuple[int, int, float, float, float]:
        width, height, focallength, zmin, zmax = _loadCameraMetadata(os.path.join(self._root, "calibration.yml"))
        return width, height, focallength, zmin, zmax

    def getCameraPoses(self):
        poses = np.array(self._poses.copy())
        orientations = R.from_matrix(poses[:, :3, :3]).as_quat()
        return poses[:, :3, 3], orientations

    def getPerturbedPoses(self, sigmaPosition, sigmaOrientation) -> Tuple[np.array, np.array]:
        poses = np.array(self._poses.copy())
        poses[..., :3, 3] += sigmaPosition*np.random.randn(poses.shape[0], 3)
        poses[..., :3, :3] += sigmaOrientation*np.random.randn(poses.shape[0], 3, 3)
        poses[..., :3, :3] = nearest_orthogonal(poses[..., :3, :3])
        return poses

    def __len__(self):
        return len(self._rgb_paths)

    def __getitem__(self, index):
        pose = self._poses[index]
        if self._invert:
            pose = np.linalg.inv(pose)

        position = pose[:3, 3]
        orientation = R.from_matrix(pose[:3, :3]).as_quat()

        rgbImage = self._loadImage(self._rgb_paths[index])
        depthImage = self._loadImage(self._depth_paths[index])
        return index, position, orientation, rgbImage[...,:3], depthImage[...,0]

class PoseDataset(th.utils.data.Dataset):

    def __init__(self, path : str, invertPose : bool, dtype = np.float32, pointset = "pointset.ply", pose = "pose.dat", verbose=True) -> None:
        super().__init__()
        self._root = path
        self._invert = invertPose
        self._dtype = dtype
        self._pointset = pointset

        # Load camera poses for all frames
        self._poses = _readPoses(os.path.join(path, pose), np.float32)

    def getPointset(self) -> Tuple[np.array, np.array, np.array]:
        return _loadPointset(os.path.join(self._root, self._pointset), np.float32)

    def getCameraMetadata(self) -> Tuple[int, int, float, float, float]:
        return _loadCameraMetadata(os.path.join(self._root, "calibration.yml"))

    def getCameraPoses(self):
        poses = np.array(self._poses.copy())
        orientations = R.from_matrix(poses[:, :3, :3]).as_quat()
        return poses[:, :3, 3], orientations

    def __len__(self):
        return len(self._poses)

    def __getitem__(self, index):
        pose = self._poses[index]
        if self._invert:
            pose = np.linalg.inv(pose)

        position = pose[:3, 3]
        orientation = R.from_matrix(pose[:3, :3]).as_quat()

        return index, position, orientation

def _computeViewFrustum(width, height, focallength, zmin, zmax):
    camPos = np.array([0.0]*3)
    camForward = np.array([0.0, 0.0, -1.0])
    camUp = np.array([0.0, 1.0, 0.0])
    camRight = np.array([1.0, 0.0, 0.0])
    viewRatio = width/height

    # Compute the near/far plane centers
    nearCenter = camPos - camForward * zmin
    farCenter = camPos - camForward * zmax

    # Compute the near/far height
    nearHeight = 2 * 1/focallength * zmin #2 * tan(fovRadians/ 2) * zmin
    farHeight = 2 * 1/focallength * zmax #2 * tan(fovRadians / 2) * zmax
    nearWidth = nearHeight * viewRatio
    farWidth = farHeight * viewRatio

    farTopLeft = farCenter + camUp * (farHeight*0.5) - camRight * (farWidth*0.5)
    farTopRight = farCenter + camUp * (farHeight*0.5) + camRight * (farWidth*0.5)
    farBottomLeft = farCenter - camUp * (farHeight*0.5) - camRight * (farWidth*0.5)
    farBottomRight = farCenter - camUp * (farHeight*0.5) + camRight * (farWidth*0.5)

    nearTopLeft = nearCenter + camUp * (nearHeight*0.5) - camRight * (nearWidth*0.5)
    nearTopRight = nearCenter + camUp * (nearHeight*0.5) + camRight * (nearWidth*0.5)
    nearBottomLeft = nearCenter - camUp * (nearHeight*0.5) - camRight * (nearWidth*0.5)
    nearBottomRight = nearCenter - camUp * (nearHeight*0.5) + camRight * (nearWidth*0.5)

    vertices = [
        nearTopLeft, nearTopRight, nearBottomRight, nearBottomLeft,
        farTopLeft, farTopRight, farBottomRight, farBottomLeft
        ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # near plane edges
        (4, 5), (5, 6), (6, 7), (7, 4), # far plane edges
        (0,4), (1, 5), (2, 6), (3, 7) # connecting edges
    ]

    faces = [(0,1,2,3), # near plane
        (0,1,5,4), # left plane
        (1,2,6,5), # top plane
        (2,6,7,3), # right plane
        (0,3,7,4), # bottom plane
        (4,5,6,7) # far plane
    ]

    return vertices, edges, faces

def _transformFrustum(pose, frustum):
        
        rotation = pose[:3,:3]
        translation = pose[3,:3]

        vertices, edges, faces = frustum
        transformedVertices = []
        for point in vertices:
            point = rotation @ point + translation
            transformedVertices.append(point)

        return transformedVertices, edges, faces

def _frustumIntersections(frustumA, frustumB):

    verticesA, edgesA, facesA = frustumA
    verticesB, edgesB, facesB = frustumB

    def _normalize(x):
        return x/np.linalg.norm(x)

    # The frustums intersect two points of an b edge are on different sizes of an a plane
    for face in facesA:

        # Plane equation for the face
        origin = verticesA[face[1]]
        point1 = verticesA[face[0]]
        point2 = verticesA[face[2]]

        tangent = _normalize(point1 - origin)
        bitangent = _normalize(point2 - origin)
        normal = _normalize(np.cross(tangent, bitangent))
        polygon = Polygon([verticesA[p] for p in face])

        for edge in edgesB:

            point1 = verticesB[edge[0]]
            point2 = verticesB[edge[1]]
            direction = _normalize(point2 - point1)

            if math.isclose(direction.dot(normal), 0.0):
                break

            distance = (origin - point1).dot(normal)/direction.dot(normal)
            intersection = Point(point1 + distance*direction)

            if polygon.contains(intersection):
                return True

    return False

class FixedViewDataloader():

    def __init__(self, path: str, targetView : int, batchSize : int, invertPose: bool, flipImage: bool, device, dtype, verbose = True) -> None:
        self._root = path
        self._target = targetView
        self.batchSize = batchSize
        self._flip = flipImage
        self._dtype = dtype
        self._device = device

        # Load camera poses for all frames
        self._poses = _readPoses(os.path.join(path, "pose.dat"), np.float32)
        if invertPose is True:
            self._poses = [np.linalg.inv(pose) for pose in self._poses]

        # Gather sorted rgb and depth frame file paths
        self._rgb_paths = _readFramePaths(path, "rgb.txt")
        self._depth_paths = _readFramePaths(path, "depth.txt")

        # Drop unused frames or poses
        limit = min([len(self._poses), len(self._rgb_paths), len(self._depth_paths)])
        self._poses = self._poses[:limit]
        self._rgb_paths = self._rgb_paths[:limit]
        self._depth_paths = self._depth_paths[:limit]

        # Define camera forward direction
        camForward = np.array([0.0, 0.0, -1.0])

        # Load camera metadata and convert them to meter
        self._width, self._height, focallength, zmin, zmax = self.getCameraMetadata()
        self._width, self._height, _ = self._loadImage(self._rgb_paths[0]).shape
        focallength = focallength/100
        frustum = _computeViewFrustum(self._width, self._height, focallength, zmin, zmax)

        self.groups = []
        self.weights = []
        self.batchLimit = len(self._poses)

        poseA = self._poses[targetView]
        forwardA = poseA[:3,:3] @ camForward
        frustumA = _transformFrustum(poseA, frustum)
        for jdx, poseB in enumerate(self._poses):

            if jdx == targetView:
                continue

            forwardB = poseB[:3,:3] @ camForward
            frustumB = _transformFrustum(poseB, frustum)

            if _frustumIntersections(frustumA, frustumB):
                cos = forwardA.dot(forwardB)
                self.groups.append(jdx)
                self.weights.append(cos)

        self.batchLimit = min(self.batchLimit, len(self.groups))
        self.weights = 0.5*(np.array(self.weights)+1)
        total = np.sum(self.weights)
        self.weights = np.array(self.weights)/total

        if verbose is True:
            print("FixedViewDataloader: Found {} candidates".format(len(self.groups)))

    def getPointset(self) -> Tuple[np.array, np.array, np.array]:
        return _loadPointset(os.path.join(self._root, "pointset.ply"), np.float32)

    def getCameraMetadata(self) -> Tuple[int, int, float, float, float]:
        return _loadCameraMetadata(os.path.join(self._root, "calibration.yml"))

    def _loadImage(self, filepath):
        image = Image.open(filepath, mode='r')
        if self._flip is True:
            image = ImageOps.flip(image)
        image = np.array(image, dtype=np.float32)/255.0
        if len(image.shape) == 3:
            image = np.transpose(image, (1,0,2))
        else:
            image = np.transpose(image, (1,0))
        return image

    def __len__(self):
        return len(self.groups)//self.batchSize

    def __iter__(self):
        self.batch = 0
        #self._indices = np.random.permutation(np.arange(len(self.groups)))
        self._indices = np.random.permutation(self.groups)
        return self

    def __next__(self):

        if (self.batch+1)*self.batchSize >= len(self.groups):
            raise StopIteration()

        #batchIndices = self._indices[self.batch*self.batchSize: (self.batch+1)*self.batchSize]
        batchIndices = np.empty(self.batchSize, dtype=np.int32)
        batchIndices[0] = self._target
        batchIndices[1:] = np.random.choice(self.groups, self.batchSize-1, p = self.weights)
        self.batch += 1

        imageTensor = th.empty(size=[self.batchSize, self._width, self._height,3])
        depthTensor = th.empty(size=[self.batchSize, self._width, self._height])
        positionTensor = th.empty(size=(self.batchSize, 3))
        orientationTensor = th.empty(size=(self.batchSize, 4))

        for idx, view in enumerate(batchIndices):
            pose = self._poses[view]
            orientation = R.from_matrix(pose[:3, :3]).as_quat()

            rgbImage = self._loadImage(self._rgb_paths[view])
            depthImage = self._loadImage(self._depth_paths[view])

            positionTensor[idx] = th.from_numpy(pose[:3, 3])
            orientationTensor[idx] = th.from_numpy(orientation)
            imageTensor[idx] = th.from_numpy(rgbImage[...,:3])
            depthTensor[idx] = th.from_numpy(depthImage)

        return positionTensor.to(self._device), orientationTensor.to(self._device), imageTensor.to(self._device), depthTensor.to(self._device)

class StructuredDataloader(th.utils.data.Dataset):

    def __init__(self, path, batchSize, invertPose, flipImage, dtype, device, verbose=True) -> None:
        super().__init__()

        self._root = path
        self.batchSize = batchSize
        self._flip = flipImage
        self._dtype = dtype
        self._device = device

        # Load camera poses for all frames
        self._poses = _readPoses(os.path.join(path, "pose.dat"), np.float32)
        if invertPose is True:
            self._poses = [np.linalg.inv(pose) for pose in self._poses]


        # Gather sorted rgb and depth frame file paths
        self._rgb_paths = _readFramePaths(path, "rgb.txt")
        self._depth_paths = _readFramePaths(path, "depth.txt")

        # Drop unused frames or poses
        limit = min([len(self._poses), len(self._rgb_paths), len(self._depth_paths)])
        self._poses = self._poses[:limit]
        self._rgb_paths = self._rgb_paths[:limit]
        self._depth_paths = self._depth_paths[:limit]

        # Define camera forward direction
        camForward = np.array([0.0, 0.0, -1.0])
        
        # Compute view frustum boundary
        self._width, self._height, focallength, zmin, zmax = self.getCameraMetadata()
        focallength = focallength/100
        frustum = _computeViewFrustum(self._width, self._height, focallength, zmin, zmax)

        # Compute visual cone intersections
        if verbose:
            print("Dataset: Computing overlaps ...")
        self.batchLimit = len(self._poses)

        self.groups = [ [] for _ in range(len(self._poses))]
        self.weights = [ [] for _ in range(len(self._poses))]
        for idx, poseA in enumerate(self._poses):
            self.groups[idx] = []
            
            forwardA = poseA[:3,:3] @ camForward
            frustumA = _transformFrustum(poseA, frustum)
            for jdx, poseB in enumerate(self._poses):

                forwardB = poseB[:3,:3] @ camForward
                frustumB = _transformFrustum(poseB, frustum)

                if _frustumIntersections(frustumA, frustumB):
                    cos = forwardA.dot(forwardB)
                    self.groups[idx].append(jdx)
                    self.weights[idx].append(cos)

            if verbose and (idx % 10 == 0):
                print("Dataset: Computing overlaps ... {}".format(idx))
        
            self.batchLimit = min(self.batchLimit, len(self.groups[idx]))
            self.weights[idx] = 0.5*(np.array(self.weights[idx])+1)
            total = np.sum(self.weights[idx])
            self.weights[idx] = np.array(self.weights[idx])/total

        if verbose is True:
            print("Dataset: found {} images".format(len(self._rgb_paths)))
            print("Dataset: Batchsize upper bound is {}".format(self.batchLimit))

        self.randperm = np.random.permutation(np.arange(len(self._poses)))

    def _loadImage(self, filepath):
        image = Image.open(filepath, mode='r')
        if self._flip is True:
            image = ImageOps.flip(image)
        image = np.array(image, dtype=np.float32)/255.0
        if len(image.shape) == 3:
            image = np.transpose(image, (1,0,2))
        else:
            image = np.transpose(image, (1,0))
        return image

    def getPointset(self) -> Tuple[np.array, np.array, np.array]:
        return _loadPointset(os.path.join(self._root, "pointset.ply"), np.float32)

    def getCameraMetadata(self) -> Tuple[int, int, float, float, float]:
        return _loadCameraMetadata(os.path.join(self._root, "calibration.yml"))

    def __len__(self):
        return len(self._poses)

    def __iter__(self):
        self.index = 0
        self.randperm = np.random.permutation(np.arange(len(self._poses)))
        return self

    def __next__(self):

        if self.index >= len(self._poses):
            raise StopIteration()

        viewIdx = self.randperm[self.index]
        self.index += 1

        indices = np.random.choice(self.groups[viewIdx], size=[self.batchSize-1], replace=False, p=self.weights[viewIdx])
        indices = list(indices) + [viewIdx]
        imageTensor = th.empty(size=[self.batchSize, self._width, self._height,3])
        depthTensor = th.empty(size=[self.batchSize, self._width, self._height])
        positionTensor = th.empty(size=(self.batchSize, 3))
        orientationTensor = th.empty(size=(self.batchSize, 4))

        for idx, index in enumerate(indices):
            pose = self._poses[index]
            orientation = R.from_matrix(pose[:3, :3]).as_quat()

            rgbImage = self._loadImage(self._rgb_paths[index])
            depthImage = self._loadImage(self._depth_paths[index])

            positionTensor[idx] = th.from_numpy(pose[:3, 3])
            orientationTensor[idx] = th.from_numpy(orientation)
            imageTensor[idx] = th.from_numpy(rgbImage[...,:3])
            depthTensor[idx] = th.from_numpy(depthImage[...,0])

        return positionTensor.to(self._device), orientationTensor.to(self._device), imageTensor.to(self._device), depthTensor.to(self._device)

if __name__ == "__main__":
 
    def quaterion2Matrix(quaterion):
        m = np.zeros(shape=[3,3])#th.zeros(size=[1, 4, 4])

        # Rotation matrix based on Euler–Rodrigues formula
        x,y,z,w = quaterion#.split(1, dim=-1)

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        m[..., 0, 0] = x2 - y2 - z2 + w2
        m[..., 1, 0] = 2 * (xy + zw)
        m[..., 2, 0] = 2 * (xz - yw)

        m[..., 0, 1] = 2 * (xy - zw)
        m[..., 1, 1] = - x2 + y2 - z2 + w2
        m[..., 2, 1] = 2 * (yz + xw)

        m[..., 0, 2] = 2 * (xz + yw)
        m[..., 1, 2] = 2 * (yz - xw)
        m[..., 2, 2] = - x2 - y2 + z2 + w2

        return m

    import matplotlib.pyplot as plt
    # path: str, targetView : int, batchSize : int, invertPose: bool, flipImage: bool, device, dtype, verbose = True
    dataset = FixedViewDataloader("../Dataset/Material23", 779, 16, False, False, th.device('cpu'), th.float32)
    for position, orientation, images, depth in dataset:
        print(images.shape)

        fig, ax = plt.subplots(4, 4)
        for idx, image in enumerate(images):
            row = idx // 4
            col = idx % 4
            print(image.shape)
            ax[row, col].imshow(image.transpose(0,1))
        plt.show()
   
