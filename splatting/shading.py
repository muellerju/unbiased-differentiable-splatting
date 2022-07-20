import torch as th
import math

from .extensions import DiffuseShading, SphericalHarmonicShading
from .extensions.utils import dot

def _cosine(normal : th.Tensor, light : th.Tensor) -> th.Tensor:
    cosAlpha = th.sum(light.mul(normal), dim=-1, keepdim=True)
    cosAlpha = th.clamp(cosAlpha, 0.0, 1.0)
    return cosAlpha

class Lambert(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ambientLight : th.Tensor, lampDirections : th.Tensor, lampIntensities : th.Tensor, normals : th.Tensor, diffuse : th.Tensor) -> th.Tensor:
        bn, l, _ = lampDirections.shape
        lampDirections = lampDirections.view(bn, l, 1, 3)
        normals = normals.unsqueeze(1)

        ambientIrradiance = ambientLight*diffuse

        bn, l, _ = lampIntensities.shape
        radiance = lampIntensities.view(bn, l, 1, 3)
        cos = _cosine(normals, lampDirections)
        irradiance = radiance.mul(cos)

        brdf = diffuse.unsqueeze(1)/math.pi
        return ambientIrradiance + th.mul(irradiance, brdf).sum(dim=1)

class DeferredLambert(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ambientLight : th.Tensor, lightDirections : th.Tensor, lampIntensities : th.Tensor, normals : th.Tensor, diffuse : th.Tensor) -> th.Tensor:
        # Expected input shapes:
        # lightDirections bn, l, 3
        # lampIntensities bn, l, 3
        # normals bn, w, h, 3
        # diffuse bn, w, h, 3

        bn, l, _ = lightDirections.shape
        lightDirections = lightDirections.view(bn, 1, 1, l, 3)
        bn, l, _ = lampIntensities.shape
        radiance = lampIntensities.view(bn, 1, 1, l, 3)
        print("radiance", radiance.shape)
        
        bn, w, h, _ = normals.shape
        normals = normals.view(bn, w, h, 1, 3)
        diffuse = diffuse.view(bn, w, h, 1, 3)

        cos = _cosine(normals, lightDirections)
        print("cos", cos.shape)
        irradiance = radiance.mul(cos)
        print("irradiance", irradiance.shape)
        brdf = diffuse #/math.pi
        print("brdf", brdf.shape)

        return th.mul(irradiance, brdf).sum(dim=3)

class DeferredSphericalHaromic(th.nn.Module):
    def __init__(self, batched=1) -> None:
        super().__init__()

        self.coefficients = th.nn.Parameter(data=th.Tensor(batched, 3, 9), requires_grad=True)
        self.coefficients.data.uniform_(-0.1, 0.1)
        self.coefficients.data[...,0] = 0.5

        self.shading = SphericalHarmonicShading.apply


    def forward(self, normals : th.Tensor, diffuse : th.Tensor) -> th.Tensor:
        # Expected input shapes:
        # normals bn, w, h, 3
        # diffuse bn, w, h, 3

        return self.shading(self.coefficients, normals, diffuse)    

class ForwardSphericalHaromic(th.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.coefficients = th.nn.Parameter(data=th.Tensor(1, 3, 9), requires_grad=True)
        self.coefficients.data.uniform_(-0.1, 0.1)
        self.coefficients.data[...,0] = 0.5

    def buildShMatrix(self):
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        M = th.empty([1, 3, 4, 4], device=self.coefficients.device, dtype=self.coefficients.dtype)
        M[..., 0, 0] = c1 * self.coefficients[..., 8]
        M[..., 1, 0] = c1 * self.coefficients[..., 4]
        M[..., 2, 0] = c1 * self.coefficients[..., 7]
        M[..., 3, 0] = c2 * self.coefficients[..., 3]

        M[..., 0, 1] = c1 * self.coefficients[..., 4]
        M[..., 1, 1] = -c1 * self.coefficients[..., 8]
        M[..., 2, 1] = c1 * self.coefficients[..., 5]
        M[..., 3, 1] = c2 * self.coefficients[..., 1]

        M[..., 0, 2] = c1 * self.coefficients[..., 7]
        M[..., 1, 2] = c1 * self.coefficients[..., 5]
        M[..., 2, 2] = c3 * self.coefficients[..., 6]
        M[..., 3, 2] = c2 * self.coefficients[..., 2]

        M[..., 0, 3] = c2 * self.coefficients[..., 3]
        M[..., 1, 3] = c2 * self.coefficients[..., 1]
        M[..., 2, 3] = c2 * self.coefficients[..., 2]
        M[..., 3, 3] = c4 * self.coefficients[..., 0] - c5 * self.coefficients[..., 6]

        return M

    def forward(self, normals : th.Tensor, diffuse : th.Tensor):

        shMatrix = self.buildShMatrix().view(1,1,3,4,4)
        #print(shMatrix.shape)
        n = th.cat([normals, th.ones([1, normals.shape[1], 1], device=diffuse.device, dtype=diffuse.dtype)], dim=-1)
        n = n.view(1, normals.shape[1], 1, 1, 4)
        #print(shMatrix.shape, n.shape)
        #print((n @ shMatrix).shape)
        image = dot(n @ shMatrix, n, False)
        #print(image.shape)
        #input()
        return diffuse*image.squeeze(-1)/math.pi
