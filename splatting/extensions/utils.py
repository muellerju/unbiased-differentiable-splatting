import torch as th
from typing import Tuple

import gc
import numpy as np

def normalize(x : th.Tensor, eps = 1e-4) -> th.Tensor:
    norm = x.norm(dim=-1, keepdim=True)
    return x/ (norm + eps)

def world2Camera(camPositions : th.Tensor, camQuaternion : th.Tensor) -> th.Tensor:
    print("World2Camera device ", camPositions.device )
    batchSize = camPositions.shape[0]
    dtype = camPositions.dtype
    device = camPositions.device
    viewMatrix = th.zeros(size=[batchSize, 1, 4, 4], dtype=dtype, device=device)

    # Rotation matrix based on Euler–Rodrigues formula
    x,y,z,w = camQuaternion.split(1, dim=-1)

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

    viewMatrix[..., 0, 0] = x2 - y2 - z2 + w2
    viewMatrix[..., 1, 0] = 2 * (xy + zw)
    viewMatrix[..., 2, 0] = 2 * (xz - yw)

    viewMatrix[..., 0, 1] = 2 * (xy - zw)
    viewMatrix[..., 1, 1] = - x2 + y2 - z2 + w2
    viewMatrix[..., 2, 1] = 2 * (yz + xw)

    viewMatrix[..., 0, 2] = 2 * (xz + yw)
    viewMatrix[..., 1, 2] = 2 * (yz - xw)
    viewMatrix[..., 2, 2] = - x2 - y2 + z2 + w2

    # Set translation part
    viewMatrix[..., :3, 3] = camPositions.unsqueeze(1)
    viewMatrix[..., 3, 3] = 1.0

    return viewMatrix

def transformForward(viewMatrix : th.Tensor, forward : str) -> th.Tensor:
    # Transformation
    if forward == 'z':
        mirror = th.eye(4).view(1,1,4,4).to(viewMatrix.device)
        mirror[...,2,2] = -1.0
        flip = th.eye(4).view(1,1,4,4).to(viewMatrix.device)
        flip[...,1,1] = -1.0
        viewMatrix = flip @ mirror @ viewMatrix

    return viewMatrix



def dot(x : th.Tensor, y : th.Tensor, keepdim : bool) -> th.Tensor:
    return th.sum(x.mul(y), dim=-1, keepdim=keepdim)

def projectOntoPlane(x : th.Tensor, camPoints : th.Tensor, camNormals : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    normalAngle = dot(camPoints, camNormals, True)
    xts = normalAngle/dot(x, camNormals, True)
    return x*xts, xts

def det22(matrices : th.Tensor) -> th.Tensor:
    return matrices[...,0,0]*matrices[...,1,1]-matrices[...,0,1]*matrices[...,1,0]

def inverse22(matrices : th.Tensor) -> th.Tensor:
    """
    computes the inverses of a list of 2x2 matrices
    """
    detInv = 1/det22(matrices)
    R = th.empty(matrices.size(), dtype=matrices.dtype, device=matrices.device)
    R[..., 0, 0] = matrices[..., 1, 1] * detInv
    R[..., 0, 1] = -matrices[..., 0, 1] * detInv
    R[..., 1, 0] = -matrices[..., 1, 0] * detInv
    R[..., 1, 1] = matrices[..., 0, 0] * detInv
    return R

def diag22(values : th.Tensor) -> th.Tensor:
    shape = list(values.shape[:-1]) + [2,2]
    dtype = values.dtype
    device = values.device
    matrices = th.zeros(size=shape, dtype=dtype, device=device) # bn, w, h, m, 2, 2
    matrices[...,0,0] = values[...,0]
    matrices[...,1,1] = values[...,1]
    return matrices

def trace22(values : th.Tensor) -> th.Tensor:
    return values[...,0,0] + values[...,1,1]

def transformToObjectspace(screenPoints : th.Tensor, scale : th.Tensor, screenSize : th.Tensor) -> th.Tensor:
    dtype = screenPoints.dtype
    device = screenPoints.device

    if len(screenPoints.shape) == 5:
        screenSize = screenSize.view(1, 1, 1, 1, 2) # bn, w, h, n, c
    else:
        screenSize = screenSize.view(1, 1, 2) # bn, n, c
    scale = scale.view_as(screenSize)

    shape = list(screenPoints.shape)
    shape[-1] += 1

    objectPoint = th.ones(size=shape, dtype=dtype, device=device)
    objectPoint[...,:2] = -(screenSize - 2.0*screenPoints)/(2.0*scale)
    return objectPoint

def camera2Screen(scale : th.Tensor) -> th.Tensor:
    dtype = scale.dtype
    device = scale.device

    projection = th.zeros(size=[1,1,4,4], dtype=dtype, device=device)
    projection[...,0,0] = scale[0]
    projection[...,1,1] = -scale[1]
    projection[...,2,2] = 1.0 #(self.zmax + self.zmin)/(self.zmax - self.zmin)
    projection[...,3,2] = -1.0

    return projection

def listAllocatedTensors():
    print("-"*80)
    print("List of allocated tensors")
    print("-"*80)
    allocatedTensors = 0
    allocatedBytes = 0
    for obj in gc.get_objects():
        try:
            if th.is_tensor(obj) or (hasattr(obj, 'data') and th.is_tensor(obj.data)):
                print(type(obj), obj.size())
                allocatedTensors += 1
                allocatedBytes += np.product(list(obj.shape))*4
        except:
            pass
    print("-"*80)
    print("Allocated tensors:", allocatedTensors)
    print("Allocated memory:", allocatedBytes, "({} gib)".format(allocatedBytes/(2**30)))
    print("-"*80)

def conditionalNormalization(dx, radius=1.0, eps = 1e-4):
    norm = dx.norm(dim=-1, keepdim=True)
    scale = th.clamp(norm + eps, 1.0)
    return dx.div(scale)
