from numpy.lib.function_base import diff
import torch as th
import math

class L2ImageLoss(th.nn.Module):

    def __init__(self, reduction : str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor):
        if self.reduction == 'mean':
            return th.square(x - y).mean(), None
        elif self.reduction == 'sum':
            return th.square(x - y).sum(), None
        else:
            raise ValueError("Unkown reduction mode")

class L1ImageLoss(th.nn.Module):

    def __init__(self, reduction : str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor):
        if self.reduction == 'mean':
            return th.abs(x - y).mean(), None
        elif self.reduction == 'sum':
            return th.abs(x - y).sum(), None
        else:
            raise ValueError("Unkown reduction mode")

class HuberImageLoss(th.nn.Module):

    def __init__(self, delta : float, reduction : str = 'mean') -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor) -> th.Tensor:
        scaling = self.delta

        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)

        if self.reduction == 'mean':
            return loss.mean(), None
        elif self.reduction == 'sum':
            return loss.sum(), None
        else:
            raise ValueError("Unkown reduction mode")

class MaskedL1Loss(th.nn.Module):

    def __init__(self, threshold : float, reduction : str = 'mean') -> None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, x, target):

        mask = (target.sum(dim=-1, keepdims=True) > self.threshold).to(target.dtype)
        loss = mask.detach()*th.abs(x-target)

        if self.reduction == 'mean':
            return th.sum(loss/mask.sum()), mask.detach()
        elif self.reduction == 'sum':
            return loss.sum(), mask.detach()
        else:
            raise ValueError("Unkown reduction mode")

class MaskedL2Loss(th.nn.Module):

    def __init__(self, threshold : float, reduction : str = 'mean') -> None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor):

        mask = (y.sum(dim=-1, keepdims=True) > self.threshold).to(y.dtype)
        loss = mask.detach()*th.square(x-y)

        if self.reduction == 'mean':
            return th.sum(loss/mask.sum()), mask.detach()
        elif self.reduction == 'sum':
            return loss.sum(), mask.detach()
        else:
            raise ValueError("Unkown reduction mode")

class MaskedL1Loss(th.nn.Module):

    def __init__(self, threshold : float, reduction : str = 'mean') -> None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor):

        mask = (y.sum(dim=-1, keepdims=True) > self.threshold).to(y.dtype)
        loss = mask.detach()* th.abs(x-y)

        if self.reduction == 'mean':
            return th.sum(loss/mask.sum()), mask.detach()
        elif self.reduction == 'sum':
            return loss.sum(), mask.detach()
        else:
            raise ValueError("Unkown reduction mode")

class MaskedHuberLoss(th.nn.Module):

    def __init__(self, threshold : float, delta : float, reduction : str = 'mean') -> None:
        super().__init__()
        self.threshold = threshold
        self.delta = delta
        self.reduction = reduction

    def forward(self, x : th.Tensor, y : th.Tensor) -> th.Tensor:
        scaling = self.delta

        mask = (y.sum(dim=-1, keepdims=True) > self.threshold).to(y.dtype)
        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
        loss = mask.detach()*loss

        if self.reduction == 'mean':
            return th.sum(loss/mask.sum()), mask.detach()
        elif self.reduction == 'sum':
            return loss.sum(), mask.detach()
        else:
            raise ValueError("Unkown reduction mode")

class MultiScalarLoss(th.nn.Module):

    def __init__(self, weights):
        super().__init__()

        self.weights = weights
        self.downsample = th.nn.AvgPool2d(2, divisor_override=1)

    def forward(self, x, target):
        x = x.permute(0,3,1,2,)
        target = target.permute(0,3,1,2)

        loss = 0
        for step, weight in enumerate(self.weights):
            loss += weight * th.abs(x - target).sum()

            x = self.downsample(x)
            target = self.downsample(target).detach()

        return loss, None

class SmoothLoss(th.nn.Module):

    def __init__(self, kernel_size, sigma, weights):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        #self.weights = weights
        self.channels = 3
        self.lossFn = HuberImageLoss(delta=0.1, reduction='sum')

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = th.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = th.stack([x_grid, y_grid], dim=-1)
        self.register_buffer('xy_grid', xy_grid)

        self.mean = (kernel_size - 1)/2.
        self.variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        """gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          th.exp(
                              -th.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / th.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)"""

        self.gaussian_filter = th.nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=kernel_size, groups=self.channels,
                                    padding=kernel_size//2, bias=False)

        #self.gaussian_filter.weight.data = gaussian_kernel
        #self.gaussian_filter.weight.requires_grad = False

        #self.smoothing = th.nn.Conv2d(3, 3, kernel_size, groups=3, padding=kernel_size//2, bias=False)
        #self.smoothing.weight.data.fill_(1/float(kernel_size**2))
        #self.smoothing.weight.requires_grad = False
        self._computeWeights()

    def _computeWeights(self):
        gaussian_kernel = (1./(2.*math.pi*self.variance)) *\
                          th.exp(
                            -th.sum((self.xy_grid - self.mean)**2., dim=-1) /\
                            (2*self.variance)
                           )
        gaussian_kernel = gaussian_kernel / th.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def decaySmoothing(self, gamma=0.5):
        self.sigma -= gamma
        self.variance = max(0.1, self.sigma)**2
        self._computeWeights()

    def forward(self, x, target):
        x = x.permute(0,3,1,2)
        target = target.permute(0,3,1,2)

        #for step, weight in enumerate(self.weights):
        #    loss += weight* th.abs(x - target.detach()).sum()

        x = self.gaussian_filter(x)
        #target = self.gaussian_filter(target)
        loss, _ = self.lossFn(x, target.detach())

        return loss, target.permute(0, 2, 3, 1)


if __name__ == "__main__":

    def huber(x, y, scaling=0.1):
        """
        A helper function for evaluating the smooth L1 (huber) loss
        between the rendered silhouettes and colors.
        """
        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
        return loss

    scaling = 0.1
    difference = th.linspace(-1, 1, steps=100)
    diff_sq = difference ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    
    print(loss.min(), loss.max())

    import matplotlib.pyplot as plt
    plt.plot(difference, loss)
    plt.plot(difference, th.square(difference))
    plt.plot(difference, th.abs(difference))
    plt.show()    
