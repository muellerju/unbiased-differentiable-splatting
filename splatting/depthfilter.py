import torch as th
from typing import Tuple

from .extensions import PixelScatterGather
from .extensions.utils import *

class AccumulativeFiltering(th.nn.Module):
    def __init__(self, opts, width, height, focallength, zmax) -> None:
        super().__init__()
        self.dtype = opts.dtype
        self.device = opts.device

        self.zmax = zmax
        self.width = width
        self.height = height
        self.screenSize = th.tensor([width, height], dtype=opts.dtype, device=opts.device)
        self.scale = th.empty(size=[2], dtype=opts.dtype, device=opts.device)
        self.scale[...,0] = focallength
        self.scale[...,1] = focallength
        self.gather_nd = PixelScatterGather.apply

    def forward(self, indices : th.LongTensor, Gk : th.Tensor, transformation : th.Tensor, positions : th.Tensor, normals : th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        bn, _, _, n,_ = Gk.shape

        _, m, _ = positions.shape
        homPoints = th.empty(size=[bn, m , 4, 1], dtype=self.dtype, device=self.device)
        homPoints[...,:3,0] = positions
        homPoints[...,3,0] = 1.0

        # Transform homogenous points and normals into camera space
        camPoints = transformation.matmul(homPoints)[...,:3,0]
        camNormals = transformation[:,:,:3,:3].matmul(normals.unsqueeze(-1)).squeeze(-1)

        camPoints = self.gather_nd(indices, camPoints)
        camNormals = self.gather_nd(indices, camNormals)
        
        # Create pixel grid
        wPixels = th.arange(self.width, dtype=self.dtype, device=self.device)
        hPixels = th.arange(self.height, dtype=self.dtype, device=self.device)
        xgrid, ygrid = th.meshgrid(wPixels, hPixels)
        #pixels = th.stack([ygrid, xgrid], dim=-1).view(1, self.width, self.height, 1, 2) #bn, w, h, m, 2
        pixels = th.stack([xgrid, ygrid], dim=-1).view(1, self.width, self.height, 1, 2) #bn, w, h, m, 2
        print(pixels[...,0].max(), pixels[...,1].max())

        # Compute depths based on the transformations
        camspacePixels = -normalize(transformToObjectspace(pixels.squeeze(-1), self.scale, self.screenSize))
        _, depths = projectOntoPlane(camspacePixels, camPoints, camNormals)
        depths[ depths < 0 ] = self.zmax
        depths[ indices < 0 ] = self.zmax

        # Sort depth
        sortedIndices = th.argsort(depths, dim=3)
        Gk = th.gather(Gk, 3, sortedIndices)
        depths = th.gather(depths, 3, sortedIndices)
        indices = th.gather(indices, 3, sortedIndices.squeeze(-1))

        # Create weight map by using multiplicative depth blending
        clampedGk = th.clamp(Gk, 0.0, 1.0)
        weights = clampedGk
        product = th.cumprod(1 - clampedGk[:,:,:,:n-1,...], dim=3)
        weights[:,:,:,1:,...] *= product
        return indices, weights, depths
