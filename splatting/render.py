import torch as th
from typing import Any, Tuple
import math

import matplotlib.pyplot as plt

from .shading import *
from .sampling import *
from .extensions import PixelScatterGather, FmmSampling, EWASplatting, DepthFiltering, DiffuseShading, AffineTransform, LinearTransform, LinearInterpolation
from .depthfilter import *
from .extensions.utils import *

from cudaExtensions import sampling as cudaSampling

class EWAWeight(th.nn.Module):

    def __init__(self, opts, width : int, height : int, focallength : float, zmin : float, zmax : float, smoothing : float) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.zmin = zmin
        self.zmax = zmax
        self.smoothing = smoothing
        self.dtype = opts.dtype
        self.device = opts.device
        
        self.screenSize = th.tensor([width, height], dtype=self.dtype, device=self.device)
        self.scale = th.empty(size=[2], dtype=self.dtype, device=self.device)
        aspect = self.width/self.height
        #fov = 2*math.atan(36.0/(2*focallength))
        #self.scale[...,0] = 1.0/(math.tan(0.5*fov)*aspect)
        #self.scale[...,1] = -1.0/math.tan(0.5*fov)
        self.scale[...,0] = focallength*aspect
        self.scale[...,1] = focallength
        self.projection = self._camera2Screen()
        self.gather_nd = PixelGather.apply
        #self.importanceSampling = NaiveSampling()
        self.importanceSampling = BatchFmmSampling()
        self.numSamples = 25

    def _camera2Screen(self) -> th.Tensor:

        projection = th.zeros(size=[1,1,4,4], dtype=self.dtype, device=self.device)
        projection[...,0,0] = self.scale[0]
        projection[...,1,1] = -self.scale[1]
        projection[...,2,2] = 1.0 #(self.zmax + self.zmin)/(self.zmax - self.zmin)
        projection[...,3,2] = -1.0

        return projection

    def _transformToObjectspace(self, screenPoints : th.Tensor) -> th.Tensor:
        if len(screenPoints.shape) == 5:
            screenSize = self.screenSize.view(1, 1, 1, 1, 2) # bn, w, h, n, c
        else:
            screenSize = self.screenSize.view(1, 1, 2) # bn, n, c
        scale = self.scale.view_as(screenSize)

        shape = list(screenPoints.shape)
        shape[-1] += 1

        objectPoint = th.ones(size=shape, dtype=self.dtype, device=self.device)
        objectPoint[...,:2] = (screenSize - 2.0*screenPoints)/(2.0*scale)
        return objectPoint

    def _buildInverseJacobian(self, screenPoints : th.Tensor, camPoints : th.Tensor, camNormals : th.Tensor) -> th.Tensor:
        # Reproject the screen space basis vectors
        y0 = screenPoints.clone()
        y0[...,0] += 1.0
        x0 = self._transformToObjectspace(y0)
        y1 = screenPoints.clone()
        y1[...,1] += 1.0
        x1 = self._transformToObjectspace(y1)
        
        # Projec onto the tangent plane
        x0Tilde, _ = projectOntoPlane(x0, camPoints, camNormals)
        x0Tilde -= camPoints
        x1Tilde, _ = projectOntoPlane(x1, camPoints, camNormals)
        x1Tilde -= camPoints
        
        # Build tangent frame
        u0 = normalize(x0Tilde)
        u1 = u0.cross(camNormals)
        
        # Construct Jacobian
        shape = list(screenPoints.shape) + [2]
        invJk = th.zeros(size=shape, dtype=self.dtype, device=self.device)
        invJk[...,0,0] = x0Tilde.norm(dim=-1)
        invJk[...,0,1] = dot(x1Tilde, u0, keepdim=False)
        invJk[...,1,1] = dot(x1Tilde, u1, keepdim=False)
        return invJk

    def _buildCovariance(self, invJk : th.Tensor, pointStdDevs : th.Tensor) -> th.Tensor:
        Jk = _inverse22(invJk)
        eye = self.smoothing * th.eye(2, dtype=self.dtype, device=self.device)
        Vrk = _diag22(pointStdDevs.square())
        if len(invJk.shape) == 6:
            eye = eye.view(1, 1, 1, 1, 2, 2) # bn, w, h, m, 2, 2
            Vk = eye + th.matmul(Jk, th.matmul(Vrk, Jk.transpose(4, 5)))
        else:
            eye = eye.view(1, 1, 2, 2) # bn, m, 2, 2
            Vk = eye + th.matmul(Jk, th.matmul(Vrk, Jk.transpose(2, 3)))
        return Vk

    def forward(self, camPoints : th.Tensor, camNormals : th.Tensor, pointStdDevs : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        bn, n, _, _ = camPoints.shape
        
        # Project points into screen space
        print(camPoints[0,0,:])
        projectedPoints = self.projection.matmul(camPoints)
        print(projectedPoints[0,0,:])
        screenPoints = projectedPoints[...,:2,0].div(projectedPoints[...,3,0].unsqueeze(-1))
        screenPoints += self.screenSize.view(1,1,2)/2
        print(screenPoints[0,0,:])
        print(projectedPoints[...,3,0].min(), projectedPoints[...,3,0].max())
        #ndcPoints = projectedPoints[...,:3,0].div(projectedPoints[...,3,0].unsqueeze(-1))
        #screenPoints = ((ndcPoints[...,:2] + 1.0)/2.0)*self.screenSize.view(1,1,2)
        #self.screenPoints = screenPoints.detach().cpu().clone()

        # Reshape points after projection
        camPoints = camPoints[:,:,:3].squeeze(-1)
        camNormals = camNormals.squeeze(-1)

        listAllocatedTensors()

        # Find an approximate standart deviation the mariginal distribution per point
        with th.no_grad():
            invJacobians = self._buildInverseJacobian(screenPoints, camPoints, camNormals)
            Vk = self._buildCovariance(invJacobians, pointStdDevs)
            marginals = 0.5*trace22(Vk)
            avgStdDeviation = marginals.mean()
            print(avgStdDeviation)
        #avgStdDeviation = 1.0

        # Perform importance sampling and gather positions and normals with relevant weights
        indices, invalid = self.importanceSampling(screenPoints, self.width, self.height, self.numSamples, stdDev=avgStdDeviation)
        camPoints = self.gather_nd(camPoints, indices)
        camNormals = self.gather_nd(camNormals, indices)
        screenPoints = self.gather_nd(screenPoints, indices)
        pointStdDevs = self.gather_nd(pointStdDevs, indices)

        self.indices = indices.detach().cpu().clone()
        #invVk = self.gather_nd(invVk, indices)
        #normalization = self.gather_nd(normalization, indices)

        # Build Jacobian and Covarianz matrix and compute gauss normalization
        invJk = self._buildInverseJacobian(screenPoints, camPoints, camNormals)
        Vk = self._buildCovariance(invJk, pointStdDevs)
        invVk = inverse22(Vk)
        #Jk = _inverse22(invJk)
        #eye = th.eye(2, dtype=self.dtype, device=self.device).view(1, 1, 1, 1, 2, 2) # bn, w, h, m, 2, 2
        #print("invJk.shape", invJk.shape)
        #print("Jk.shape", Jk.shape)
        #print("eye.shape", eye.shape)
        #print("diag22.shape", _diag22(pointStdDevs).shape)
        #Vk = Jk.matmul(_diag22(pointStdDevs)).matmul(Jk.transpose(4,5)) + self.smoothing * eye
        #normalization = 2*math.pi*th.sqrt(_det22(Vk))*th.abs(_det22(invJk)) #th.ones([bn,n,1])#
        #invVk = _inverse22(Vk)
        #print("Covariance Finite check", th.isfinite(invVk).all())

        # Create pixel grid
        wPixels = th.arange(self.width, dtype=self.dtype, device=self.device)
        hPixels = th.arange(self.height, dtype=self.dtype, device=self.device)
        xgrid, ygrid = th.meshgrid(wPixels, hPixels)
        pixels = th.stack([ygrid, xgrid], dim=-1).view(1, self.width, self.height, 1, 2) #bn, w, h, m, 2

        # Compute gauss weights per pixel
        print("pixels.shape", pixels.shape, "screenPoints.shape", screenPoints.shape)
        x = th.unsqueeze(pixels - screenPoints, dim=-1)
        exponent = -0.5* x.transpose(4,5).matmul(invVk).matmul(x)
        normalization = 2*math.pi*th.sqrt(det22(Vk))*th.abs(det22(invJk))
        Gk = th.exp(exponent.squeeze(-1))/normalization.unsqueeze(-1)
        
        # Compute depth values per pixel and order relevant indices according to their depth (per pixel)
        camspacePixels = -normalize(self._transformToObjectspace(pixels.squeeze(-1)))
        _, depths = projectOntoPlane(camspacePixels, camPoints, camNormals)
        depths[ depths < 0 ] = self.zmax
        depths[ invalid ] = self.zmax

        return indices, Gk, depths

def _normalizeWeights(weights : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    print("Weights shape", weights.shape)
    accumulated = weights.sum(3, keepdim=True)
    accumulated = th.where(accumulated < 0.01, th.ones_like(accumulated), accumulated)
    nWeights = weights/accumulated
    return nWeights, accumulated

def _normalizeNormals(normals : th.Tensor) -> th.Tensor:
    norms = normals.norm(dim=-1, keepdim=True)
    normals = th.where(norms < 0.01, th.ones_like(normals), normals)
    return normalize(normals)

class BackfaceCulling(th.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, normals, camNormals):

        normalAngle = camNormals[...,2].unsqueeze(-1)
        
        normals = th.where(normalAngle > 0, -normals, normals)
        camNormals = th.where(normalAngle > 0, -camNormals, camNormals)

        return normals, camNormals

'''class LinearInterpolation(th.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, attribute : th.Tensor, weights : th.Tensor) -> th.Tensor:
        return th.sum(weights.mul(attribute), dim=3)

class SphericalInterpolation(th.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, attribute : th.Tensor, weights : th.Tensor) -> th.Tensor:
        weights = weights.squeeze(-1)
        theta = th.acos(attribute[...,2])
        phi = th.asin(attribute[...,1]/th.sin(theta))
        interTheta = th.sum(weights.mul(theta), dim=3)
        interPhi = th.sum(weights.mul(phi), dim=3)
        
        interpolatedAttribute = th.stack([th.sin(interTheta)*th.cos(interPhi), th.sin(interTheta)*th.sin(interPhi), th.cos(interTheta)], dim=-1)
        return normalize(interpolatedAttribute)'''

class Splatting(th.nn.Module):

    def __init__(self, opts, logger, width : int, height : int, focallength : float, zmin : float, zmax : float, smoothing : float) -> None:
        super().__init__()
        self.logger = logger
        
        # Intrinsic camera parameter
        self.width = width
        self.height = height
        self.focallength = focallength

        # Coordinate system parameter
        self.coordForward = opts.forward

        # Rendering parameters
        self.smoothing = smoothing
        self.numSamples = opts.numSamples
        self.precision = opts.precision
        self.shadingMode = opts.shadingMode
        
        # Technical parameter
        self.dtype = opts.dtype
        self.device = opts.device

        # Intermediate modules and autograd functions
        self.affineTransform = AffineTransform.apply
        self.linearTransform = LinearTransform.apply
        self.backface = BackfaceCulling()
        self.sampling = FmmSampling()
        self.ewa = EWASplatting.apply
        self.depthBlending = DepthFiltering.apply
        self.linearInterpolation = LinearInterpolation.apply
        self.deferredShading = DiffuseShading.apply
        self.alphaGather = PixelScatterGather.apply
        self.forwardShading = Lambert()
        self.sphericalHarmonics = DeferredSphericalHaromic()
        self.forwardHarmonics = ForwardSphericalHaromic()

    def forward(self,
        camPositions : th.Tensor, camOrientations : th.Tensor, 
        pointPositions : th.Tensor, pointNormals : th.Tensor, pointStdDevs : th.Tensor, pointDiffuse : th.Tensor, 
        ambientLight : th.Tensor, lampDirections : th.Tensor, lampIntensities : th.Tensor,
        alpha : th.Tensor = None) -> th.Tensor:

        bn = camPositions.shape[0]
        _, n, _ = pointPositions.shape
        #if camPositions.dim() == 3:
        #    transformMatrix = camPositions.contiguous()
        #else:
        transformMatrix = world2Camera(camPositions, camOrientations) #.squeeze(1)
        #transformMatrix = th.linalg.inv(transformMatrix.squeeze(1)).unsqueeze(1)
        transformMatrix = transformForward(transformMatrix, self.coordForward).squeeze(1)

        # Transform point positions and normals into camera space
        camPoints = self.affineTransform(transformMatrix, pointPositions)
        camNormals = self.linearTransform(transformMatrix, pointNormals)

        # Orient all normals to face the camera
        pointNormals, backfaceNormals = self.backface(pointNormals, camNormals)

        # Perform per pixel importance sampling
        with th.no_grad():
            indices = self.sampling(
                self.width, self.height, self.focallength,
                camPoints, backfaceNormals, pointDiffuse, pointStdDevs,
                self.numSamples, self.smoothing, self.precision)
            self.indices = indices.cpu().clone()

        # Compute screen space weights
        ewaWeights = self.ewa(
            self.width, self.height, self.focallength,
            indices, camPoints, camNormals, pointStdDevs,
            self.smoothing)

        # Compute weights
        #if alpha is not None:
        #    print(alpha.shape)
        #    perPixelAlpha = self.alphaGather(indices, alpha)
        #    print(ewaWeights.shape, perPixelAlpha.shape)
        #    ewaWeights = ewaWeights*perPixelAlpha[...,0]
        #    print(ewaWeights.shape)

        # Perform the depth filtering step
        depthWeights = self.depthBlending(
            indices, ewaWeights,
            self.width, self.height, self.focallength,
            camPoints, camNormals)

        # Transform light directions into camera space
        camLampDirections = self.linearTransform(transformMatrix, lampDirections)

        # Evluate shading model
        if self.shadingMode == "deferred":

            # Interpolate attribute of sampled points with depth filtered ewa weights
            ewaNormals = _normalizeNormals(self.linearInterpolation(indices, depthWeights, camNormals))
            #print("ewaNormals", ewaNormals.shape, ewaNormals.min(), ewaNormals.max())
            ewaDiffuse = self.linearInterpolation(indices, depthWeights, pointDiffuse)
            #print("ewaDiffuse", ewaDiffuse.shape, ewaDiffuse.min(), ewaDiffuse.max())

            # Store filtered attribute functions
            #self.logger.logNormals(ewaNormals, "normals")
            #self.logger.logImages(ewaDiffuse, [0, 1], "diffuse")

            image = self.deferredShading(ambientLight, camLampDirections, lampIntensities, ewaNormals, ewaDiffuse)

        elif self.shadingMode == "forward":

            shadedPoints = self.forwardShading(ambientLight, camLampDirections, lampIntensities, camNormals, pointDiffuse)
            image = self.linearInterpolation(indices, depthWeights, shadedPoints)

        elif self.shadingMode == "sphericalHarmonics":

            # Interpolate attribute of sampled points with depth filtered ewa weights
            ewaNormals = self.linearInterpolation(indices, depthWeights, pointNormals)
            #print("ewaNormals", ewaNormals.shape, ewaNormals.min(), ewaNormals.max())
            ewaDiffuse = self.linearInterpolation(indices, depthWeights, pointDiffuse)
            #print("ewaDiffuse", ewaDiffuse.shape, ewaDiffuse.min(), ewaDiffuse.max())

            # Store filtered attribute functions
            #self.logger.logNormals(ewaNormals, "normals")
            #self.logger.logImages(ewaDiffuse, [0, 1], "diffuse")

            image = self.sphericalHarmonics(ewaNormals, ewaDiffuse)

        elif self.shadingMode == "forwardHarmonics":

            shadedPoints = self.forwardHarmonics(pointNormals, pointDiffuse)
            image = self.linearInterpolation(indices, depthWeights, shadedPoints)

        elif self.shadingMode == "None":

            image = self.linearInterpolation(indices, depthWeights, pointDiffuse)

        elif self.shadingMode == "silhouette":

            image = th.sum(ewaWeights, dim=-1, keepdim=True)
            image = th.clamp(image, 0, 1)

        else:
            raise ValueError("Unkown shading mode")

        #listAllocatedTensors()

        # Evalute the shading model per pixel
        #print("image", image.shape, image.min(), image.max())
        return image, indices, depthWeights

    def computeCovariance(self, camPositions : th.Tensor, camOrientations : th.Tensor, pointPositions : th.Tensor, pointNormals : th.Tensor, pointColors : th.Tensor, pointStdDevs : th.Tensor):
        bn, _ = camPositions.shape
        _, n, _ = pointPositions.shape
        transformMatrix = world2Camera(camPositions, camOrientations).squeeze(1)

        # Transform point positions and normals into camera space
        camPoints = self.affineTransform(transformMatrix, pointPositions)
        camNormals = self.linearTransform(transformMatrix, pointNormals)

        # Orient all normals to face the camera
        pointNormals, camNormals = self.backface(pointNormals, camNormals)

        # Compute current covariance matrices
        covariances = cudaSampling.covariance(
            self.width, self.height, self.focallength, 
            camPoints, camNormals, pointStdDevs, 
            self.smoothing
        )

        return camPoints, covariances

    def computeFMMWeights(self, camPositions : th.Tensor, camOrientations : th.Tensor, pointPositions : th.Tensor, pointNormals : th.Tensor, pointColors : th.Tensor, pointStdDevs : th.Tensor):
        bn, _ = camPositions.shape
        _, n, _ = pointPositions.shape
        transformMatrix = world2Camera(camPositions, camOrientations, self.coordForward).squeeze(1)

        # Transform point positions and normals into camera space
        camPoints = self.affineTransform(transformMatrix, pointPositions)
        camNormals = self.linearTransform(transformMatrix, pointNormals)

        # Orient all normals to face the camera
        pointNormals, camNormals = self.backface(pointNormals, camNormals)

        fmmWeights = sampling.approximateFMMWeights(
            self.width, self.height, self.focallength,
            camPoints, camNormals, pointColors, pointStdDevs,
            self.smoothing, self.precision
        )

        return fmmWeights

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    points = th.normal(0.0, 1.0, size=[1,100,3])
    points = normalize(points)
    normals = points.detach().clone()
    stdDevs = 0.0001*th.ones([1,100,2])

    camPosition = th.FloatTensor([0.0, 0.0, 2.0]).unsqueeze(0)
    camOrientations = th.FloatTensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0)

    red = th.FloatTensor([1.0, 0.0, 0.0]).view(1,1,3)
    blue = th.FloatTensor([0.0, 0.0, 1.0]).view(1,1,3)
    interploate = 0.5*(points[...,2].unsqueeze(-1)+1) 
    diffuse = (1-interploate)*red + interploate*blue#th.ones(size=[1,100,3])
    lampIntensities = th.ones(size=[1,10,3])
    lampDirections = normalize(th.normal(0.0, 1.0, size=(1,10,3)))

    fov = math.pi/2
    renderer = Splatting(128, 128, fov, 0.1, 50.0) #EWAWeight(128, 128, math.pi/2, 0.1, 50.0)
    image, camPoints, camNormals, pixelWeights, pixelDiffuse, pixelNormals = renderer(camPosition, camOrientations, points, normals, stdDevs, diffuse, lampDirections, lampIntensities)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    for point, color in zip(points[0], diffuse[0]):
        ax.scatter(point[0], point[1], point[2], c=[color.numpy()])
    #ax = fig.add_subplot(122)
    #ax.scatter(screen[:,0], screen[:,1])
    #ax.set_xlim(0, 128)
    #ax.set_ylim(0, 128)

    fig = plt.figure(1)
    ax = fig.add_subplot(221)
    ax.set_title("EWA Weights")
    ax.imshow(pixelWeights.sum(1).squeeze())
    ax = fig.add_subplot(222)
    ax.set_title("Diffuse")
    ax.imshow(pixelDiffuse.squeeze())
    ax = fig.add_subplot(223)
    ax.set_title("Normal")
    ax.imshow(pixelNormals.squeeze())

    fig = plt.figure(2)
    plt.imshow(image.squeeze())

    plt.show()
