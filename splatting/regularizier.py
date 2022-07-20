from math import dist
import torch as th
from torch._C import Value
from .extensions import PixelScatterGather, LinearInterpolation
from cudaExtensions import pixelgather, nearestsamples
import time
from .extensions.utils import normalize


def _normalize(x, eps=0.001):
    return x/(x.norm(dim=-1, keepdim=True)+eps)

def _dot(x, y):
    return th.sum(x*y, dim=-1, keepdim=True)

class NonNegativeRegularizer(th.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.relu = th.nn.ReLU()

    def forward(self, inputs):
        if self.reduction == "sum":
            return self.relu(-inputs).sum()
        elif self.reduction == "mean":
            return self.relu(-inputs).mean()
        else:
            raise ValueError("Unkown reduction mode")

class DensityRegularizer(th.nn.Module):

    def __init__(self, threshold, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

        self.gather =  PixelGather.apply
        self.interpolate = LinearInterpolation.apply


    def forward(self, weights, indices, points):

        sampledPoints = self.gather(points, indices) # bn, w, h, 25, 3
        targetPoints = self.interpolate(indices, weights, points).unsqueeze(3) # bn, w, h, 1, 3
        distance = (targetPoints.detach() - sampledPoints).norm(dim=-1)
        mask = (distance < self.threshold).to(distance.dtype).unsqueeze(-1)
        loss = mask.detach()*distance

        if self.reduction == "sum":
            loss = loss.sum(dim=[1,2,3])
            return loss.mean()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError("Unkown reduction mode")

class SurfaceRegularizer(th.nn.Module):

    def __init__(self, threshold, reduction = 'mean'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

        self.gather =  PixelScatterGather.apply
        self.interpolate = LinearInterpolation.apply

    def forward(self, indices, weights, points, normals):
        sampledPoints = self.gather(indices, points)

        with th.no_grad():
            surfacePoints = self.interpolate(indices, weights, points).unsqueeze(3)
            surfaceNormals = _normalize(self.interpolate(indices, weights, normals)).unsqueeze(3)
            distance = (surfacePoints.detach() - sampledPoints).norm(dim=-1)
            mask = (distance < self.threshold).to(points.dtype).unsqueeze(-1)

        #print(th.isfinite(surfaceNormals).all())
        #print(mask.shape, sampledPoints.shape, surfacePoints.shape, surfaceNormals.shape)
        loss = mask.detach()*_dot(sampledPoints - surfacePoints.detach(), surfaceNormals.detach())
        #print(loss.shape)
        #input()

        if self.reduction == "sum":
            loss = loss.sum(dim=[1,2,3])
            return loss.mean()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError("Unkown reduction mode")

class InvDecayRegularizer(th.nn.Module):

    def __init__(self, threshold, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

        self.gather = pixelgather.gather

    def forward(self, indices, weights, points):
        # Compute mask based on the weights
        with th.no_grad():
            accumulatedWeights = self.gather(indices, weights.unsqueeze(-1), points.shape[1]) # bn,w,h,k,1 -> bn,n,1
            mask = (accumulatedWeights < self.threshold).to(points.dtype).sum(dim=0, keepdim=True) # bn, n, 1 -> 1, n, 1
            mask = -th.clamp(mask, 0, 1)

        # Inv weight decay
        # TODO: Die Multiplizität sollte berücksichtigt werden?
        loss = mask.detach()*points.norm(dim=-1, keepdim=True)

        # Perform loss reduction
        if self.reduction == "sum":
            loss = loss.sum(dim=[1,2,3])
            return loss.mean()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError("Unkown reduction mode")

class NearestSamplesRegularizer(th.nn.Module):

    def __init__(self, sigma=0.5, num_nearest = 10, reduction = "sum") -> None:
        super().__init__()
        self.sigma = sigma
        self.num_nearest = num_nearest
        self.reduction = reduction

        self.nearest = nearestsamples.nearest

    def _gaussian(self, x):
        return th.exp(-x.square()/self.sigma**2)

    def forward(self, indices, weights, points, normals):

        with th.no_grad():
            #start = time.time()
            nearestIndices = self.nearest(indices, weights, points, self.num_nearest)
            #print("Nearest took", time.time()-start, "seconds.")

            mask = (nearestIndices > 0).to(points.dtype).unsqueeze(-1)
            nearestIndices.clamp_(0)
            nearestPoints = points[0, nearestIndices[0].long(), :].unsqueeze(0)
            nearestNormals = normals[0, nearestIndices[0].long(), :].unsqueeze(0) # 1, n, m, 3

            anisotropic = (normals.unsqueeze(2) - nearestNormals).norm(dim=-1, keepdim=True)
            isotropic = (points.unsqueeze(2) - nearestPoints.detach()).norm(dim=-1, keepdim=True)
            weights = mask*self._gaussian(anisotropic)*self._gaussian(isotropic)
            invWeights = 1.0/(weights.sum(dim=2)+1e-4)

            filteredNormals = normalize(th.sum(weights*nearestNormals, dim=2))
            pixelMask = mask.sum(dim=2).clamp(0, 1)
            #assert(th.isfinite(filteredNormals).all())
            #print(weights.shape, nearestNormals.shape, filteredNormals.shape)
            #input()

        # bn, n, 3 x bn, n, 7, 3
        planeDistance = weights.detach()*_dot(
            points.unsqueeze(2) - nearestPoints.detach(),
            nearestNormals.detach()
        )
        planeLoss = th.square(invWeights.detach()*planeDistance.sum(dim=2))
        #print(planeDistance.shape, planeLoss.shape)
        #input()

        normalDistance = th.square(filteredNormals.detach() - normals).sum(dim=-1, keepdim=True)
        #print(normalDistance.shape, pixelMask.shape)
        normalLoss = pixelMask.detach()*normalDistance
        #print(normalDistance.shape, mask.shape, normalLoss.shape)
        #input()

        if self.reduction == "sum":
            return normalLoss.sum(), planeLoss.sum()
        elif self.reduction == "mean":
            return normalLoss.mean(), planeLoss.mean()
        else:
            raise ValueError("Unkown reduction mode")

if __name__ == "__main__":

    value = -th.rand(size=[5])
    value.requires_grad = True
    optim = th.optim.SGD([value], lr=0.1)
    lossFn = NonNegativeRegularizer()
    for i in range(100):
        print(value)
        optim.zero_grad()
        loss = lossFn(value)
        loss.backward()
        optim.step()
