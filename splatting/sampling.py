from typing import Any
import torch as th
import math

class UniformSampling():

    @th.no_grad()
    def __call__(self, screenpoints : th.Tensor, width : int, height : int, numSamples : int, stdDev : float) -> th.LongTensor:
        device = screenpoints.device
        dtype = screenpoints.dtype
        bn, n, _ = screenpoints.shape

        index = th.arange(n, device=device)
        samples = th.randperm(index, device=device)[:numSamples]
        samples = samples.expand(bn, width, height, n)
        self.indices = samples.detach().clone().cpu()
        return samples

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

class NaiveDuplicateSampling():

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

        # Compute naive gaussian weights
        print("pixels.shape", pixels.shape, screenpoints.shape)
        x = pixels - screenpoints.view(bn, 1, 1, n, 2, 1) #bn, w, h, m, 2, 1
        exponent = -0.5* x.transpose(4,5).matmul(x)
        Gk = th.exp(exponent.squeeze(-1))/2*math.pi

        # Sort gaussian weights
        sortedIndices = th.argsort(Gk, dim=3, descending=True)

        sortedIndices = sortedIndices[:,:,:,:5,0]
        duplicateIndices = th.zeros(size=(bn, width, height, 5), dtype=th.long)
        duplicateIndices[:,:,:,:5] = sortedIndices

        invalid = duplicateIndices < 0
        duplicateIndices[invalid] = 0

        
        self.indices = duplicateIndices.detach().clone().cpu()
        return duplicateIndices.contiguous(), invalid

try:
    from fmmSampling import batchedSampling
    class BatchFmmSampling():

        @th.no_grad()
        def __call__(self, sources : th.Tensor, width : int, height : int, numSamples : int, stdDev : float = 1.0, precision : float = 0.001) -> th.LongTensor:
            bn, n, c = sources.shape
            dtype = sources.dtype
            device = sources.device
            swapedSources = th.empty(size=(bn, n, c), dtype=dtype, device=device)
            swapedSources[:,:,0] = sources[:,:,1]
            swapedSources[:,:,1] = sources[:,:,0]
            
            indices = batchedSampling(width, height, numSamples, precision, swapedSources, stdDev)
            self.indices = indices.detach().clone().cpu()

            invalid = indices < 0
            indices[invalid] = 0

            return indices.type(th.int64), invalid
except:
    print("WARNING: Using importance sampling fallback.")
    class BatchFmmSampling(NaiveSampling):
        pass
