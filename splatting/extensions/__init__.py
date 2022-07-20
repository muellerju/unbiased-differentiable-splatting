import torch as th
from typing import Tuple
import math
import timeit

from .utils import *

benchmark = True
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        if not benchmark:
            ts = time()
            result = f(*args, **kw)
            te = time()
            print('func: {} took: {} sec'.format(f.__qualname__, te-ts))
            return result
        else:
            return f(*args, **kw)
    return wrap

class NaiveGather(th.autograd.Function):

    @staticmethod
    @timing
    def forward(ctx, src : th.Tensor, indices : th.LongTensor) -> th.Tensor:
        bn, n, channels = src.shape
        dtype = src.dtype
        device = src.device

        channel = src.shape[2:]
        bn, width, height, m = indices.shape
        ctx.save_for_backward(indices)
        ctx.src_shape = [bn, width, height, n, *channel]
        ctx.indices_shapeIn = [bn, width, height, n] + [1]*len(channel)
        ctx.indices_shapeOut = [bn, width, height, m] + list(channel)

        if src.shape[0] < indices.shape[0]:
            shape = list(src.shape)
            shape[0] = indices.shape[0]
            src = src.expand(shape)

        out = th.empty([bn, width, height, m, *channel], dtype=dtype, device=device)
        #for i in range(bn):
        #    out[i] = src[i, indices[i],...]
        for b in range(bn):
            for x in range(width):
                for y in range(height):
                    for i in range(m):
                        index = indices[b, x, y, i]
                        out[b, x, y, i, :] = src[b, index, :]
        return out

    @staticmethod
    @timing
    def backward(ctx, grad_output):
        dtype = grad_output.dtype
        device = grad_output.device

        indices = ctx.saved_tensors[0]
        grad_scatter = th.zeros(*ctx.src_shape, dtype=dtype, device=device)
        #indices = indices.view(*ctx.indices_shapeIn)
        #indices = indices.expand(ctx.indices_shapeOut)
        
        grad_scatter.scatter_(3, indices.unsqueeze(-1), grad_output)
        accumulated_grad = grad_scatter.sum(dim=(1,2))
        return accumulated_grad, None

class NaiveSampling():

    @th.no_grad()
    def __call__(self, screenpoints : th.Tensor, width : int, height : int, numSamples : int, stdDev : float = 1.0) -> th.LongTensor:
        device = screenpoints.device
        dtype = screenpoints.dtype
        bn, n, _ = screenpoints.shape

        # Create pixel grid
        wPixels = th.arange(width, dtype=dtype, device=device)
        hPixels = th.arange(height, dtype=dtype, device=device)
        xgrid, ygrid = th.meshgrid(wPixels, hPixels)
        pixels = th.stack([ygrid, xgrid], dim=-1).view(1, width, height, 1, 2, 1) #bn, w, h, m, 2, 1
        print(pixels[...,0,0].max(), pixels[...,1,0].max())

        # Compute naive gaussian weights
        print("pixels.shape", pixels.shape, screenpoints.shape)
        x = pixels - screenpoints.view(bn, 1, 1, n, 2, 1) #bn, w, h, m, 2, 1
        exponent = -0.5* x.transpose(4,5).matmul(x)
        Gk = th.exp(exponent.squeeze(-1))/2*math.pi

        # Sort gaussian weights
        sortedIndices = th.argsort(Gk, dim=3, descending=True)
        print("sortedIndices.shape", sortedIndices.shape)
        sampledIndices = sortedIndices[:,:,:,:numSamples,0].contiguous()
        self.indices = sampledIndices.detach().clone().cpu()

        invalid = sampledIndices < 0
        sampledIndices[invalid] = 0

        return sampledIndices, invalid

try:
    from cudaExtensions import sampling
    class FmmSampling():

        @th.no_grad()
        def __call__(self, 
            width : int, height : int, focallength : float, 
            points : th.Tensor, normals : th.Tensor, colors : th.Tensor, stdDevs : th.Tensor, 
            numSamples : int, 
            smoothing : float, precision : float = 0.001) -> th.Tensor:

            indices = sampling.sample(width, height, focallength, points, normals, colors, stdDevs, numSamples, smoothing, precision)

            return indices
except:
    print("WARNING: Using importance sampling fallback.")
    class FmmSampling(NaiveSampling):
        pass

try:
    from cudaExtensions import transform
    class AffineTransform(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, transformMatrix, vector):

            transformedVector = transform.affineForward(transformMatrix, vector)
            ctx.save_for_backward(transformMatrix, vector)

            return transformedVector

        @staticmethod
        @timing
        def backward(ctx, gradIn):
            
            transformMatrix, vector = ctx.saved_variables
            gradTransformMatrix, gradVector = transform.affineBackward(gradIn, transformMatrix, vector)

            return gradTransformMatrix, gradVector


    class LinearTransform(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, transformMatrix, vector):
            
            transformedVector = transform.linearForward(transformMatrix, vector)
            ctx.save_for_backward(transformMatrix, vector)

            return transformedVector

        @staticmethod
        @timing
        def backward(ctx, gradIn):
            
            transformMatrix, vector = ctx.saved_variables
            gradTransformMatrix, gradVector = transform.linearBackward(gradIn, transformMatrix, vector)

            return gradTransformMatrix, gradVector
except:
    print("WARNING: Using transformation fallback!")
    class AffineTransform():
        pass

    class LinearTransform():
        pass

try:
    from cudaExtensions import splatting
    class EWASplatting(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, 
            width : int, height : int, focallength : float, 
            indices : th.Tensor,
            points : th.Tensor, 
            normals : th.Tensor, 
            stdDevs : th.Tensor, 
            smoothing : float ) -> th.Tensor:
            
            weights = splatting.forward(
                width, height, focallength,
                indices,
                points, normals, stdDevs,
                smoothing)

            ctx.save_for_backward(indices, points, normals, stdDevs)
            ctx.width = width
            ctx.height = height
            ctx.focallength = focallength
            ctx.smoothing = smoothing
            
            return weights

        @staticmethod
        @timing
        def backward(ctx, gradWeights):

            #print(gradWeights.min(), gradWeights.max())
            #input()

            indices, points, normals, stdDevs = ctx.saved_variables

            gradPoints, gradNormals, gradStdDevs = splatting.backward(
                ctx.width, ctx.height, ctx.focallength,
                gradWeights.contiguous(), indices,
                points, normals, stdDevs,
                ctx.smoothing)

            if not benchmark:
                print("EWASplatting.backward: gradPoints", gradPoints.shape, gradPoints.min(), gradPoints.max())

            gradPoints = conditionalNormalization(gradPoints)
            gradNormals = conditionalNormalization(gradNormals)
            gradStdDevs = conditionalNormalization(gradStdDevs)

            gradPoints[th.isnan(gradPoints)] = 0.0
            gradNormals[th.isnan(gradNormals)] = 0.0
            gradStdDevs[th.isnan(gradStdDevs)] = 0.0

            if not benchmark:
                print("EWASplatting.backward: gradWeights", gradWeights.shape, gradWeights.min(), gradWeights.max())
                print("EWASplatting.backward: normalized gradPoints", gradPoints.min(), gradPoints.max())
                print("EWASplatting.backward: normalized gradNormals", gradNormals.min(), gradNormals.max())
                print("EWASplatting.backward: normalized gradStdDevs", gradStdDevs.min(), gradStdDevs.max())

            return None, None, None, None, gradPoints, None, gradStdDevs, None
except:
    print("WARNING: Using importance splatting fallback.")
    class EWASplatting():

        @staticmethod
        def apply(transformations : th.Tensor, 
            width : int, height : int, focallength : float, 
            points : th.Tensor, 
            normals : th.Tensor, 
            stdDevs : th.Tensor, 
            numSamples : int, precision : float):
            bn, n, _ = points.shape
            dtype = points.dtype
            device = points.device

            homPoints = th.empty(size=[bn, n , 4, 1], dtype=dtype, device=device)
            homPoints[...,:3,0] = points
            homPoints[...,3,0] = 1.0

            # Transform homogenous points and normals into camera space
            camPoints = transformations.matmul(homPoints)
            camNormals = transformations[:,:,:3,:3].matmul(normals.unsqueeze(-1)).squeeze(-1)

            return EWASplatting.forward(width, height, focallength, numSamples, camPoints, camNormals, stdDevs)

        @staticmethod
        def buildInverseJacobian(scale : th.Tensor, screenSize : th.Tensor, screenPoints : th.Tensor, camPoints : th.Tensor, camNormals : th.Tensor) -> th.Tensor:
            dtype = camPoints.dtype
            device = camPoints.device

            # Reproject the screen space basis vectors
            y0 = screenPoints.clone()
            y0[...,0] += 1.0
            x0 = transformToObjectspace(y0, scale, screenSize)
            y1 = screenPoints.clone()
            y1[...,1] += 1.0
            x1 = transformToObjectspace(y1, scale, screenSize)
            
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
            invJk = th.zeros(size=shape, dtype=dtype, device=device)
            invJk[...,0,0] = x0Tilde.norm(dim=-1)
            invJk[...,0,1] = dot(x1Tilde, u0, keepdim=False)
            invJk[...,1,1] = dot(x1Tilde, u1, keepdim=False)
            return invJk

        @staticmethod
        def buildCovariance(invJk : th.Tensor, pointStdDevs : th.Tensor, smoothing : float) -> th.Tensor:
            dtype = invJk.dtype
            device = invJk.device

            Jk = inverse22(invJk)
            eye = smoothing * th.eye(2, dtype=dtype, device=device)
            Vrk = diag22(pointStdDevs.square())
            if len(invJk.shape) == 6:
                eye = eye.view(1, 1, 1, 1, 2, 2) # bn, w, h, m, 2, 2
                Vk = eye + th.matmul(Jk, th.matmul(Vrk, Jk.transpose(4, 5)))
            else:
                eye = eye.view(1, 1, 2, 2) # bn, m, 2, 2
                Vk = eye + th.matmul(Jk, th.matmul(Vrk, Jk.transpose(2, 3)))
            return Vk

        @staticmethod
        def forward(width : int, height : int, focallength : float, numSamples : int, camPoints : th.Tensor, camNormals : th.Tensor, pointStdDevs : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
            dtype = camPoints.dtype
            device = camPoints.device

            # Create projection matrix
            screenSize = th.tensor([width, height], dtype=dtype, device=device)
            scale = th.empty(size=[2], dtype=dtype, device=device)
            aspect = width/height
            scale[...,0] = focallength*aspect
            scale[...,1] = focallength
            projection = camera2Screen(scale)
            
            # Project points into screen space
            projectedPoints = projection.matmul(camPoints)
            screenPoints = projectedPoints[...,:2,0].div(projectedPoints[...,3,0].unsqueeze(-1))
            screenPoints += screenSize.view(1,1,2)/2
            
            # Reshape points after projection
            camPoints = camPoints[:,:,:3].squeeze(-1)
            camNormals = camNormals.squeeze(-1)

            # Find an approximate standart deviation the mariginal distribution per point
            with th.no_grad():
                invJacobians = EWASplatting.buildInverseJacobian(scale, screenSize, screenPoints, camPoints, camNormals)
                Vk = EWASplatting.buildCovariance(invJacobians, pointStdDevs, smoothing=0.1)
                marginals = 0.5*trace22(Vk)
                avgStdDeviation = marginals.mean()
                print(avgStdDeviation)
            #avgStdDeviation = 1.0

            # Perform importance sampling and gather positions and normals with relevant weights
            importanceSampling = BatchFmmSampling()
            indices, invalid = importanceSampling(screenPoints, width, height, numSamples, stdDev=avgStdDeviation)
            gather_nd = PixelGather.apply
            camPoints = gather_nd(camPoints, indices)
            camNormals = gather_nd(camNormals, indices)
            screenPoints = gather_nd(screenPoints, indices)
            pointStdDevs = gather_nd(pointStdDevs, indices)

            # Build Jacobian and Covarianz matrix and compute gauss normalization
            invJk = EWASplatting.buildInverseJacobian(scale, screenSize, screenPoints, camPoints, camNormals)
            Vk = EWASplatting.buildCovariance(invJk, pointStdDevs, smoothing=0.1)
            invVk = inverse22(Vk)

            # Create pixel grid
            wPixels = th.arange(width, dtype=dtype, device=device)
            hPixels = th.arange(height, dtype=dtype, device=device)
            xgrid, ygrid = th.meshgrid(wPixels, hPixels)
            pixels = th.stack([ygrid, xgrid], dim=-1).view(1, width, height, 1, 2) #bn, w, h, m, 2

            # Compute gauss weights per pixel
            print("pixels.shape", pixels.shape, "screenPoints.shape", screenPoints.shape)
            x = th.unsqueeze(pixels - screenPoints, dim=-1)
            exponent = -0.5* x.transpose(4,5).matmul(invVk).matmul(x)
            normalization = 2*math.pi*th.sqrt(det22(Vk))*th.abs(det22(invJk))
            Gk = th.exp(exponent.squeeze(-1))/normalization.unsqueeze(-1)
            
            # Compute depth values per pixel and order relevant indices according to their depth (per pixel)
            #camspacePixels = -normalize(self._transformToObjectspace(pixels.squeeze(-1)))
            #_, depths = projectOntoPlane(camspacePixels, camPoints, camNormals)
            #depths[ depths < 0 ] = self.zmax
            #depths[ invalid ] = self.zmax

            return indices, Gk

try:
    from  cudaExtensions import filtering
    class DepthFiltering(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, indices : th.Tensor, weights : th.Tensor, width : int, height : int, focallength : float, points : th.Tensor, normals : th.Tensor) -> th.Tensor:
            
            ctx.save_for_backward(indices, weights, points, normals)
            ctx.width = width
            ctx.height = height
            ctx.focallength = focallength

            return filtering.forward(indices, weights, width, height, focallength, points, normals)

        @staticmethod
        @timing
        def backward(ctx, gradIn : th.Tensor):

            indices, weights, points, normals = ctx.saved_variables

            gradWeights = filtering.backward(indices, weights, gradIn, ctx.width, ctx.height, ctx.focallength, points, normals)

            if not benchmark:
                print("DepthFiltering.backward: weights", weights.min(), weights.max())
                print("DepthFiltering.backward: gradIn", gradIn.min(), gradIn.max())
                print("DepthFiltering.backward: gradWeights", gradWeights.min(), gradWeights.max())

            return None, gradWeights, None, None, None, None, None, None
except:
    print("WARNING: Using per depth filtering fallback!")
    class DepthFiltering():
        pass

try:
    from cudaExtensions import pixelgather
    class PixelScatterGather(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, indices, values):
            
            bn, n, channels = values.shape
            bn, width, height, m = indices.shape
            
            ctx.save_for_backward(indices)
            ctx.numSources = n
            
            gathered = pixelgather.scatter(indices, values)

            if not benchmark:
                print("PixelGather input shapes", values.shape, indices.shape)
                print("PixelGather index range", indices.min(), indices.max())
                print("PixelGahter gathered.shape=", gathered.shape)

            return gathered

        @staticmethod
        @timing
        def backward(ctx, grad_output):

            indices = ctx.saved_tensors[0]
            accumulated_grad = pixelgather.gather(indices, grad_output, ctx.numSources)

            if not benchmark:
                print("PixelGather backward: accumulated_grad.shape", accumulated_grad.shape)

            return None, accumulated_grad
except:
    print("WARNING: Using per pixel gahter fallback!")
    class PixelScatterGather(NaiveGather):
        pass

try:
    from cudaExtensions.interpolation import forward, backward
    class LinearInterpolation(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, indices, weights, values):
           
            ctx.save_for_backward(indices, weights, values)
            interpolated = forward(indices, weights, values)

            if not benchmark:
                print("LinearInterpolation input shapes", values.shape, indices.shape)
                print("LinearInterpolation forward: weights", weights.shape, weights.min(), weights.max())
                print("LinearInterpolation forward: values", values.shape, values.min(), values.max())
                print("LinearInterpolation interpolated.shape=", interpolated.shape)

            return interpolated

        @staticmethod
        @timing
        def backward(ctx, gradIn):
        
            indices, weights, values = ctx.saved_tensors


            if not benchmark:
                print("LinearInterpolation backward: gradIn", gradIn.shape, gradIn.min(), gradIn.max())
                print("LinearInterpolation backward: weights", weights.shape, weights.min(), weights.max())
                print("LinearInterpolation backward: values", values.shape, values.min(), values.max())

            gradWeights, gradValues = backward(gradIn, indices, weights, values)

            if not benchmark:
                print("LinearInterpolation backward: gradIn", gradIn.shape, gradIn.min(), gradIn.max())
                print("LinearInterpolation backward: values", values.shape, values.min(), values.max())
                print("LinearInterpolation backward: gradWeights.shape", gradWeights.shape, gradWeights.min(), gradWeights.max())
                print("LinearInterpolation backward: gradValues.shape", gradValues.shape, gradValues.min(), gradValues.max())
            
            return None, gradWeights, gradValues
except:
    print("WARNING: Using per interpolation fallback!")
    class LinearInterpolation():
        pass

try:
    from cudaExtensions import splatting
    class NaiveSplatting(th.autograd.Function):

            @staticmethod
            @timing
            def forward(ctx, 
                transformations : th.Tensor, 
                width : int, height : int, focallength : float, 
                points : th.Tensor) -> th.Tensor:
                
                image = splatting.naiveForward(transformations, width, height, focallength, points)

                ctx.transformsShape = transformations.shape
                ctx.pointsShape = points.shape
                
                return image.unsqueeze(-1)

            @staticmethod
            @timing
            def backward(ctx, gradIndices, gradWeights):
                dtype = gradWeights.dtype
                device = gradWeights.device

                gradTransformations = th.zeros(ctx.transformsShape, dtype=dtype, device=device)
                gradPoints = th.zeros(ctx.pointsShape, dtype=dtype, device=device)

                return gradTransformations, None, None, None, gradPoints
except:
    print("WARNING: NaiveSplatting is not installed")
    class NaiveSplatting():
        pass

try:
    from cudaExtensions import shading
    class DiffuseShading(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, directions, intensites, normals, diffuse) -> th.Tensor:
           
            ctx.save_for_backward(directions, intensites, normals, diffuse)

            images = shading.forward(directions, intensites, normals, diffuse)
            return images

        @staticmethod
        @timing
        def backward(ctx, gradIn):
            
            directions, intensites, normals, diffuse = ctx.saved_variables
            gradDirections, gradIntensities, gradNormals, gradDiffuse = shading.backward(
                gradIn, directions, intensites, normals, diffuse
            )

            if not benchmark:
                print("DiffuseShading.backward: gradIn", gradIn.shape, gradIn.min(), gradIn.max())
                print("DiffuseShading.backward: gradDirections", gradDirections.min(), gradDirections.max())
                print("DiffuseShading.backward: gradIntensities", gradIntensities.min(), gradIntensities.max())
                print("DiffuseShading.backward: gradNormals", gradNormals.min(), gradNormals.max())
                print("DiffuseShading.backward: gradDiffuse", gradDiffuse.min(), gradDiffuse.max())

            return gradDirections, gradIntensities, gradNormals, gradDiffuse

    class SphericalHarmonicShading(th.autograd.Function):

        @staticmethod
        @timing
        def forward(ctx, coefficients, normals, diffuse) -> th.Tensor:
           
            ctx.save_for_backward(coefficients, normals, diffuse)

            images = shading.forwardSh(coefficients, normals, diffuse)
            return images

        @staticmethod
        @timing
        def backward(ctx, gradIn):
            
            coefficients, normals, diffuse = ctx.saved_variables

            gradCoefficients, gradNormals, gradDiffuse = shading.backwardSh(
                gradIn, coefficients, normals, diffuse
            )

            if not benchmark:
                print("SphericalHarmonics.backward: gradIn", gradIn.shape, gradIn.min(), gradIn.max())
                print("SphericalHarmonics.backward: gradCoefficients", gradCoefficients.min(), gradCoefficients.max())
                print("SphericalHarmonics.backward: gradNormals", gradNormals.min(), gradNormals.max())
                print("SphericalHarmonics.backward: gradDiffuse", gradDiffuse.min(), gradDiffuse.max())

            return gradCoefficients, gradNormals, gradDiffuse
except:
    print("WARNING: Using importance shading fallback.")
    class DiffuseShading():
        pass

    class SphericalHarmonicShading():
        pass
