import torch as th
import numpy as np
import math
import argparse
import shutil
import os
from functools import partial

import splatting.logging as logging
import splatting.losses as losses 

import splatting.render as render
import splatting.dataset as datasets
import splatting.regularizier as regularizier
from utils.math import normalize, dot
from cudaExtensions import pixelgather

class ConvBlock(th.nn.Module):

    def __init__(self, in_features, out_features, kernel_size, stride=1, norm_fn = th.nn.BatchNorm2d, activation_fn = th.nn.ReLU, drop_rate=-1) -> None:
        super().__init__()

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=kernel_size//2))
        if norm_fn is not None:
            self.layers.append(norm_fn(out_features))
        if drop_rate > 0:
            self.layers.append(th.nn.Dropout(drop_rate, True))
        if activation_fn is not None:
            self.layers.append(activation_fn())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResBlock(th.nn.Module):

    def __init__(self, num_features, activation_fn=th.nn.ReLU) -> None:
        super().__init__()

        self.layers = th.nn.ModuleList()
        self.layers.append(ConvBlock(num_features, num_features, 3, activation_fn=activation_fn))
        self.layers.append(ConvBlock(num_features, num_features, 3, activation_fn=None))

    def forward(self, x):
        skip = x.clone()
        for layer in self.layers:
            x = layer(x)
        return th.add(x, skip)

class ShuffleBlock(th.nn.Module):

    def __init__(self, num_features, upsample=2, activation_fn = th.nn.ReLU) -> None:
        super().__init__()

        in_features = num_features
        out_features = num_features*(upsample**2)

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Conv2d(in_features, out_features, 3, padding=1))
        self.layers.append(th.nn.PixelShuffle(upsample))
        self.layers.append(activation_fn())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class UnetBlock(th.nn.Module):

    def __init__(self, in_features, out_features, kernel_size = 3, norm_fn = th.nn.InstanceNorm2d, activation_fn=th.nn.ReLU, drop_rate=-1) -> None:
        super().__init__()

        self.conv1 = ConvBlock(in_features, out_features, kernel_size, norm_fn=norm_fn, activation_fn=activation_fn)
        self.conv2 = ConvBlock(out_features, out_features, kernel_size, norm_fn=norm_fn, activation_fn=activation_fn, drop_rate=drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Downsample(th.nn.Module):

    def __init__(self, num_features, kernel_size = 3) -> None:
        super().__init__()

        self.layers = th.nn.ModuleList()
        #self.layers.append(th.nn.MaxPool2d(kernel_size))
        self.layers.append(th.nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size//2, stride=2))
        self.layers.append(th.nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Upsample(th.nn.Module):

    def __init__(self, num_features, kernel_size = 3) -> None:
        super().__init__()

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.ConvTranspose2d(num_features, num_features//2, kernel_size, stride=2, padding=kernel_size//2, output_padding =1))
        self.layers.append(th.nn.ReLU())
        #self.upsample = th.nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
        #self.conv = th.nn.Conv2d(num_features, num_features//2, kernel_size, padding=kernel_size//2)

    def forward(self, x, skip):
        for layer in self.layers:
            x = layer(x)
        return th.cat([x, skip], dim=1)

class UNet(th.nn.Module):

    def __init__(self, depth, in_features, features, final=True) -> None:
        super().__init__()

        self.final = final
        instNorm = th.nn.InstanceNorm2d
        leakyReLU = partial(th.nn.LeakyReLU, negative_slope=0.2, inplace=True)
        ReLU = partial(th.nn.ReLU, inplace=True)


        # Encoder blocks
        self.encoder = th.nn.ModuleList()
        self.encoder.append(UnetBlock(in_features, features, norm_fn=None, activation_fn=leakyReLU))
        self.encoder.append(Downsample(features))
        for i in range(depth):
            num_features = features*2**i
            self.encoder.append(UnetBlock(num_features, 2*num_features, norm_fn=instNorm, activation_fn=leakyReLU))
            self.encoder.append(Downsample(2*num_features))
        self.encoder = self.encoder[:-1]

        num_features = features*2**depth
        self.intermediate = UnetBlock(num_features, num_features, norm_fn=instNorm, activation_fn=leakyReLU, drop_rate=0.5)

        # Decoder blocks
        self.decoder = th.nn.ModuleList()
        for i in range(depth):
            num_features = features*2**(depth-i)
            self.decoder.append(Upsample(num_features))
            if i < 2:
                self.decoder.append(UnetBlock(num_features, num_features//2, norm_fn=instNorm, activation_fn=ReLU, drop_rate=0.5))
            else:
                self.decoder.append(UnetBlock(num_features, num_features//2, norm_fn=instNorm, activation_fn=ReLU))

        # Final residual block
        if self.final:
            self.compress = th.nn.Conv2d(features, 3, 1)
            self.tanh = th.nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        skip = []
        for layer in self.encoder:
            x = layer(x)
            if not isinstance(layer, Downsample):
                skip.append(x)
        skip = skip[:-1]

        # Intermediate block
        x = self.intermediate(x)

        # Decoder block 1
        skip.reverse()
        for i, layer in enumerate(self.decoder):
            if not isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = layer(x, skip[i//2])

        # Final decoder level
        out = x
        if self.final:
            out = self.compress(out)
            out = self.tanh(out)
            out = out.permute(0, 2, 3, 1)

        return out 

class DropoutUNet(UNet):

    def __init__(self, depth, in_features, features, final=True) -> None:
        super().__init__(depth, in_features, features, final=final)

        self.final = final
        instNorm = th.nn.InstanceNorm2d
        leakyReLU = partial(th.nn.LeakyReLU, negative_slope=0.2, inplace=True)
        ReLU = partial(th.nn.ReLU, inplace=True)


        # Encoder blocks
        self.encoder = th.nn.ModuleList()
        self.encoder.append(ConvBlock(in_features, features, 3, norm_fn=None, activation_fn=leakyReLU))
        self.encoder.append(Downsample(features))
        for i in range(depth):
            num_features = features*2**i
            self.encoder.append(ConvBlock(num_features, 2*num_features, 3, norm_fn=instNorm, activation_fn=leakyReLU))
            self.encoder.append(Downsample(2*num_features))
        self.encoder = self.encoder[:-1]

        num_features = features*2**depth
        self.intermediate = ConvBlock(num_features, num_features, 3, norm_fn=instNorm, activation_fn=leakyReLU, drop_rate=0.5)

        # Decoder blocks
        self.decoder = th.nn.ModuleList()
        for i in range(depth):
            num_features = features*2**(depth-i)
            self.decoder.append(Upsample(num_features))
            if i < 2:
                self.decoder.append(ConvBlock(num_features, num_features//2, 3, norm_fn=instNorm, activation_fn=ReLU, drop_rate=0.5))
            else:
                self.decoder.append(ConvBlock(num_features, num_features//2, 3, norm_fn=instNorm, activation_fn=ReLU))

        # Final residual block
        if self.final:
            self.compress = th.nn.Conv2d(features, 3, 1)
            self.tanh = th.nn.Tanh()

class UpsampleUNet(th.nn.Module):

    def __init__(self, depth, in_features, features) -> None:
        super().__init__()

        self.unet = UNet(depth, in_features, features, final=False)

        # Upsampling and compression
        self.layers  = th.nn.ModuleList()
        # Upsampling component
        self.layers.append(th.nn.ConvTranspose2d(features, features, 3, stride=2, padding=1, output_padding=1))
        self.layers.append(th.nn.InstanceNorm2d(features))
        self.layers.append(th.nn.ReLU(inplace=True))
        # Final compression
        self.layers.append(th.nn.Conv2d(features, 3, 7, padding=3))
        self.layers.append(th.nn.Tanh())

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)

        x = self.unet(x)
        for layer in self.layers:
            x = layer(x)
        
        return x.permute(0, 2, 3, 1)

def initialize_conv(m):
    gain = th.nn.init.calculate_gain("relu")

    if isinstance(m, th.nn.Conv2d):
        stdv = 1. / math.sqrt(m.weight.size(1))
        th.nn.init.xavier_normal_(m.weight, gain)
        m.bias.data.uniform_(-stdv, stdv)

class Generator(th.nn.Module):
    
    def __init__(self, depth) -> None:
        super().__init__()
        activation_Fn = partial(th.nn.LeakyReLU, negative_slope=0.2)
        #activation_Fn = th.nn.PReLU partial(th.nn.LeakyReLU, negative_slope=0.2)

        # Residual blocks
        self.convBlock1 = ConvBlock(4, 64, 9, norm_fn=None, activation_fn=activation_Fn)
        self.resBlocks = th.nn.ModuleList()
        for i in range(depth):
            self.resBlocks.append(ResBlock(64, activation_fn=activation_Fn))
        self.convBlock2 = ConvBlock(64, 64, 3, activation_fn=None)

        # Upsampling block
        self.upsampleLayers = th.nn.ModuleList()
        self.upsampleLayers.append(ShuffleBlock(64, activation_fn=activation_Fn))
        self.upsampleLayers.append(ConvBlock(64, 64, 3, activation_fn=th.nn.PReLU))
        self.upsampleLayers.append(th.nn.Conv2d(64, 3, 9, padding=4))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.convBlock1(x)
        skip = x.clone()
        for layer in self.resBlocks:
            x = layer(x)
        x = self.convBlock2(x)
        x = th.add(x, skip)

        for layer in self.upsampleLayers:
            x = layer(x)
        return x.permute(0, 2, 3, 1)

class Discriminator(th.nn.Module):

    def __init__(self, depth, in_features, num_features) -> None:
        super().__init__()
        kernelSize = 4
        instNorm = th.nn.InstanceNorm2d
        leakyReLU = partial(th.nn.LeakyReLU, negative_slope=0.2, inplace=True)

        self.layers = th.nn.ModuleList()
        self.layers.append(ConvBlock(in_features, num_features, kernel_size=3, stride=2, norm_fn=None, activation_fn=leakyReLU))
        # Downsample layers
        for i in range(depth-1):
            nf = min(2*num_features, 512)
            self.layers.append(ConvBlock(num_features, nf, kernelSize, stride=2, norm_fn=instNorm, activation_fn=leakyReLU))
            num_features = nf
        # Classification layer
        nf = min(2*num_features, 512)
        self.layers.append(ConvBlock(num_features, nf, kernelSize, stride=1, norm_fn=instNorm, activation_fn=leakyReLU))
        self.layers.append(th.nn.Conv2d(nf, 1, kernelSize))
        
    def forward(self, x, prior):
        x = th.cat([x, prior], dim=-1)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers:
            x = layer(x)
                
        return x

def computeGradientPenalty(model, real, fake):
    bn = real.shape[0]

    # Compute interplation between samples
    alpha = th.rand((bn, 1, 1, 1), device=real.device, dtype=real.dtype)
    interpolated = (alpha * real + (1-alpha) * fake).requires_grad_(True)

    # Compute gradient on the interpolation
    logits = model(interpolated)
    grad_outputs = th.autograd.Variable(th.ones_like(logits).requires_grad_(False))
    gradients = th.autograd.grad(
        outputs=logits,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute gradient penalty
    slopes = th.sqrt(th.sum(th.square(gradients), dim=1))
    gradientPenalty = th.mean((slopes-1.)**2)
    return gradientPenalty

def computeSmoothnessPenalty(model, images):
    images = images.requires_grad_()
    perturbed = images + th.randn_like(images)

    # Compute gradient on the interpolation
    logits = model(perturbed)
    grad_outputs = th.autograd.Variable(th.ones_like(logits).requires_grad_(False))
    gradients = th.autograd.grad(
        outputs=logits,
        inputs=perturbed,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute gradient penalty
    slopes = th.sqrt(th.sum(th.square(gradients), dim=-1))
    gradientPenalty = th.mean((slopes-1.)**2)
    return gradientPenalty

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # Dataset parameter
        parser.add_argument("input", type=str)
        parser.add_argument("validation", type=str)
        parser.add_argument("--flipImage", type=bool, default=False)
        parser.add_argument("--invertPose", type=bool, default=True)
        parser.add_argument("--forward", type=str, default='-z')
        parser.add_argument("--output", type=str, default='./Results')
        parser.add_argument("--priorRes", type=int, default=-1)
        parser.add_argument("--imageRes", type=int, default=-1)
        parser.add_argument("--scale", type=float, default=1.0)

        # General trainings parameters
        parser.add_argument("--epochs", type=int, default=250)
        parser.add_argument("--batch", type=int, default=16)

        # Optimization parameters
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--decay", type=float, default=0.870551)
        parser.add_argument("--decaySteps", type=int, default=50)

        # Distribution regularization parameters
        parser.add_argument("--regNormal", type=float, default=-1.0)
        parser.add_argument("--regSurface", type=float, default=-1.0)
        parser.add_argument("--regSigma", type=float, default=0.5)
        parser.add_argument("--regNearest", type=int, default=10)

        # Occluded point regularization parameters
        parser.add_argument("--occlusionStep", type=float, default=0.1)
        parser.add_argument("--threshold", type=float, default=1.0)
        parser.add_argument("--visibilityDecay", type=float, default=0.0)

        # Rendering parameters
        parser.add_argument("--smoothing", type=float, default=0.0)
        parser.add_argument("--stdDevBound", type=str, default="[0.01, 0.0133]")
        parser.add_argument("--numSamples", type=int, default=40)
        parser.add_argument("--precision", type=float, default=0.01)
        parser.add_argument("--shadingMode", type=str, default="forward")
        parser.add_argument("--lightMode", type=str, default="fixed")

        # Model paramters
        parser.add_argument("--genDepth", type=int, default=4)
        parser.add_argument("--genInDim", type=int, default=16)
        parser.add_argument("--genFeatures", type=int, default=48)
        parser.add_argument("--discDepth", type=int, default=3)
        parser.add_argument("--discFeatures", type=int, default=64)

        # Technical parameters
        parser.add_argument("--dtype", type=str, default="float")
        parser.add_argument("--device", type=str, default='cuda:0')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        args.stdDevBound = list(map(float, args.stdDevBound.strip('[]').split(',')))

        if args.dtype == "float":
            args.dtype = th.float32
        else:
            raise ValueError("Unknown dtype")
        args.device = th.device(args.device)
        return args

if __name__ == "__test__":

    #upsample = UpsampleUNet(4, 16, 64)
    #print(upsample)
    #y = upsample(th.rand(16, 256, 256, 16))

    unet = UNet(5, 3, 64)
    print(unet)
    y = unet(th.rand(16, 256, 256, 3))

    inputs = th.rand(16, 256, 256, 3)
    computeSmoothnessPenalty(unet, inputs.requires_grad_())

    dropunet = DropoutUNet(5, 8, 32)
    print(dropunet)
    y = dropunet(th.rand(16, 256, 256, 8))

    disc = Discriminator(3, 4+3, 64)
    print(disc)
    logits = disc(y, th.rand(16, 256, 256, 4))

if __name__ == "__main__":

    # Parse the arguments and setup logging
    args = ArgParser().parse()
    logger = logging.TrainingLogger(args)

    # Load and setup the dataset
    dataset = datasets.Dataset(args.input, invertPose=args.invertPose, flipImage = args.flipImage, scale=args.imageRes)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    validation = datasets.Dataset(args.validation, invertPose=args.invertPose, flipImage = args.flipImage, scale=args.imageRes)
    dataloader_valid = th.utils.data.DataLoader(validation, batch_size=args.batch, shuffle=False, drop_last=False)

    vertices, normals, _ = dataset.getPointset()
    features = np.random.normal(0.0, 0.1, size=(vertices.shape[0], args.genInDim))
    vertices, normals, features, stdDevs = datasets.setupTrainablePointset(args.scale*vertices, normals, features, [0.001, 0.003], dtype=args.dtype, device=args.device)

    width, height, focallength, zmin, zmax = dataset.getCameraMetadata()
    if args.priorRes > -1:
        focallength *= (args.priorRes/width)
        width = args.priorRes
        height = args.priorRes

    print(width, height, focallength)
    input()
    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)
    generator = UNet(args.genDepth, args.genInDim, args.genFeatures).to(args.device)
    #generator = DropoutUNet(args.genDepth, args.genInDim, args.genFeatures).to(args.device)
    generator.train()
    discInDim = args.genInDim + 3
    discriminator = Discriminator(args.discDepth, discInDim, args.discFeatures).to(args.device)
    discriminator.train()

    ambientLight, lampDirections, lampIntesities = datasets.loadSceneLighting(args.input, args.dtype, args.device)

    gen_optimizer = th.optim.Adam([features] + list(generator.parameters()), lr=args.lr, betas=(0.0, 0.9))
    gen_scheduler = th.optim.lr_scheduler.StepLR(gen_optimizer, step_size=args.decaySteps, gamma=args.decay)
    
    disc_optimizer = th.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    disc_scheduler = th.optim.lr_scheduler.StepLR(disc_optimizer, step_size=args.decaySteps, gamma=args.decay)

    #advLoss_fn = th.nn.BCELoss()
    contentLoss_fn = losses.L1ImageLoss(reduction='mean')
    #contentLoss_fn = th.nn.MSELoss()

    iteration = 0
    for epoch in range(args.epochs):
        logger.incrementEpoch()

        for batchIdx, (viewIndices, cameraPosition, cameraOrientation, rgbImage, depthImage) in enumerate(dataloader):
            logger.incrementBatch()

            # Push data to the correct device
            cameraPosition = cameraPosition.to(args.device)
            cameraOrientation = cameraOrientation.to(args.device)
            rgbImage = rgbImage.to(args.device)
            depthImage = depthImage.to(args.device)

            ###
            #  Prior image and generator
            ###
            print(cameraPosition.shape, cameraOrientation.shape)
            gen_optimizer.zero_grad()
            generator.zero_grad()
            prior, _, depths = renderer(cameraPosition, cameraOrientation, vertices, normals, stdDevs, features, ambientLight, lampDirections, lampIntesities)
            fake = generator(prior)

            ###
            # Train discriminator
            ###
            disc_optimizer.zero_grad()
            discriminator.zero_grad()

            real_logits = discriminator(rgbImage, prior.detach())
            errD_real = 0.5*th.mean(th.square(real_logits-1))
            errD_real.backward()
            
            fake_logits = discriminator(fake.detach(), prior.detach())
            errD_fake = 0.5*th.mean(th.square(fake_logits))
            errD_fake.backward()

            errD = errD_real + errD_fake
            disc_optimizer.step()
            th.cuda.empty_cache()

            ###
            # Train generator
            ###
            fake_logits = discriminator(fake, prior.detach())
            advLoss = 0.5*th.mean(th.square(fake_logits-1))
            cntLoss = contentLoss_fn(fake, rgbImage)[0]
            errG = 100.0*cntLoss + advLoss #contentLoss_fn(fake, rgbImage)[0] + 0.5*th.mean(th.square(fake_logits-1))
            errG.backward()
            th.cuda.empty_cache()

            smoothLoss = computeSmoothnessPenalty(generator, prior.detach())
            smoothLoss.backward()

            gen_optimizer.step()
            th.cuda.empty_cache()

            # Enforce constraints on shading model
            with th.no_grad():
                normals.set_(normalize(normals))
                stdDevs.clamp_(*args.stdDevBound)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Disc-loss: {:.6f} Gen-loss: {:.6f} Cnt-loss: {:.6f} Smooth-loss: {:.6f}'.format(
                epoch, batchIdx * len(rgbImage), len(dataloader.dataset),
                100. * batchIdx / len(dataloader), errD.item(), errG.item(), cntLoss.item(), smoothLoss.item()
                ))

            # Write loss and gradient informations
            logger.logLosses([errD, errG, cntLoss, smoothLoss], gen_optimizer.param_groups[0]['lr'])
            logger.logGradient(features.grad, "features")

            # Clean up memory after logging gradients
            generator.zero_grad()
            discriminator.zero_grad()
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            th.cuda.empty_cache()


            # Store samples as an image
            logger.logViewIndices(viewIndices)
            iteration += 1

        # Reposition points which no longer contribute to the image
        """with th.no_grad():
            # Gather indices of non-contributing points
            nonContributingIndices = th.nonzero(colors.sum(dim=-1).squeeze() < args.threshold).squeeze()
            contributingIndices = th.nonzero(colors.sum(dim=-1).squeeze() >= args.threshold).squeeze()

            if (nonContributingIndices.dim() > 0 and nonContributingIndices.shape[0] > 0) and False:
                logger.logRepositioning(nonContributingIndices)

                # Select new position for occluded points
                randomIndices = th.randint(high=contributingIndices.shape[0], size = nonContributingIndices.shape, device=args.device)
                newIndices = contributingIndices[randomIndices]

                # Compute a randomized offset
                offset = th.randn((newIndices.shape[0], 3), dtype=args.dtype, device=args.device)
                offset = vertices[0,newIndices,:] + args.occlusionStep*normalize(offset)

                # Project offset point into the plane of the selected point
                orthogonal = dot(offset - vertices[0,newIndices,:], normals[0, newIndices, :])
                newPosition = offset - orthogonal * normals[0, newIndices, :]

                # Reposition non-contributing points
                vertices[0, nonContributingIndices, :] = newPosition #vertices[0, visibleIndices,:]
                normals[0, nonContributingIndices, :] = normals[0, newIndices, :]
                colors[0, nonContributingIndices, :].fill_(1.0)"""

        # Update scheduler after each epoch
        gen_scheduler.step()
        disc_scheduler.step()
        print("Scheduler: Updated shading lr to", gen_optimizer.param_groups[0]['lr'])

        # Save scene parameters after each epoch
        if epoch % 100 == 0:
            logger.logPointset(vertices, normals, th.ones_like(vertices))
            logger.logStdDevs(stdDevs)
            logger.logFeatures(features)
            logger.logLighting(renderer, lampDirections, lampIntesities)        
            logger.logNeuralState(generator, gen_optimizer, "generator")
            logger.logNeuralState(discriminator, disc_optimizer, "discriminator")

        if epoch % 100 == 0:
            print("Computing validation loss... ", end="")
            generator.eval()
            validLogger = open(os.path.join(logger.basepath, "validation.csv") , mode="a")
            with th.no_grad():
                for _, validPosition, validOrientation, validImage, _ in dataloader_valid:
                    validPosition = validPosition.to(args.device)
                    validOrientation = validOrientation.to(args.device)
                    validImage = validImage.to(args.device)

                    prior, _, depths = renderer(validPosition, validOrientation, vertices, normals, stdDevs, features, ambientLight, lampDirections, lampIntesities)
                    validFake = generator(prior)

                    validLoss = contentLoss_fn(validFake, validImage)[0]
                    validLogger.write(str(validLoss.item()) + ", ")

                    logger.logImages(validFake, [0,1], "neural")
            validLogger.write("\n")
            validLogger.close()
            generator.train()
            print("Done")

    logger.logPointset(vertices, normals, th.ones_like(vertices))
    logger.logStdDevs(stdDevs)
    logger.logFeatures(features)
    logger.logLighting(renderer, lampDirections, lampIntesities)        
    logger.logNeuralState(generator, gen_optimizer, "generator")
    logger.logNeuralState(discriminator, disc_optimizer, "discriminator")
    logger.close()


    # Perform a simple out-lier removale step and copy data to be able to render the dataset
    shutil.copy(os.path.join(args.input, "calibration.yml"), os.path.join(logger.basepath, "calibration.yml"))
    shutil.copy(os.path.join(args.input, "pose.dat"), os.path.join(logger.basepath, "pose.dat"))

    infile = os.path.join(logger.basepath, f"pointset_epoch{args.epochs}.ply")
    outfile = os.path.join(logger.basepath, "pointset.ply")
    shutil.copy(infile, outfile)

    infile = os.path.join(logger.basepath, f"features_epoch{args.epochs}.npy")
    outfile = os.path.join(logger.basepath, "features.ply")
    shutil.copy(infile, outfile)

    shutil.copy(os.path.join(logger.basepath, f"lights_epoch{args.epochs}.json"), os.path.join(logger.basepath, "lights.json"))
    shutil.copy(os.path.join(logger.basepath, f"neural_generator_{args.epochs}.pt"), os.path.join(logger.basepath, "neural_generator.pt"))
