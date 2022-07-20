from ast import parse
from PIL.Image import FASTOCTREE
import torch as th
import numpy as np
import math
import argparse
import shutil
import os
import open3d as o3d
from torch._C import device
from plyfile import PlyData, PlyElement

import splatting.logging as logging
import splatting.losses as losses 

import splatting.render as render
import splatting.dataset as datasets
import splatting.regularizier as regularizier
from utils.math import normalize, dot
from cudaExtensions import pixelgather
from render_dataset import generateLights

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # Dataset parameter
        parser.add_argument("input", type=str)
        parser.add_argument("--flipImage", type=bool, default=False)
        parser.add_argument("--invertPose", type=bool, default=True)
        parser.add_argument("--forward", type=str, default='z')
        parser.add_argument("--output", type=str, default='./Results')

        # General trainings parameters
        parser.add_argument("--steps", type=int, default=10)

        # Pose optimization parameters
        parser.add_argument("--poseEpochs", type=int, default=20)
        parser.add_argument("--poseLr", type=float, default=0.001)
        parser.add_argument("--poseDecay", type=float, default=0.5)
        parser.add_argument("--poseDecaySteps", type=int, default=10)

        # Scene optimization parameters
        parser.add_argument("--sceneEpochs", type=int, default=20)
        parser.add_argument("--sceneBatch", type=int, default=16)
        parser.add_argument("--sceneLr", type=float, default=0.001)
        parser.add_argument("--sceneDecay", type=float, default=0.5)
        parser.add_argument("--sceneDecaySteps", type=int, default=10)

        # Geometry optimization parameter
        parser.add_argument("--optimizePosition", type=bool, default=False)
        parser.add_argument("--optimizeAlpha", type=bool, default=False)
        parser.add_argument("--geometryEpochs", type=int, default=20)
        parser.add_argument("--geometryLr", type=float, default=0.001)
        parser.add_argument("--geometryDecay", type=float, default=0.5)
        parser.add_argument("--geometryDecaySteps", type=int, default=10)

        # Noise filter points
        parser.add_argument("--probabilistFilter", type=bool, default=False)
        parser.add_argument("--errorMean", type=float, default=0.1)
        parser.add_argument("--rejectionProb", type=float, default=1e-3)

        # Rendering parameters
        parser.add_argument("--smoothing", type=float, default=0.0)
        parser.add_argument("--stdDevBound", type=str, default="[0.001, 0.001]")
        parser.add_argument("--numSamples", type=int, default=40)
        parser.add_argument("--precision", type=float, default=0.01)
        parser.add_argument("--shadingMode", type=str, default="sphericalHarmonics")
        parser.add_argument("--lightMode", type=str, default="fixed")

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

if __name__ == "__main__":

    # Parse the arguments and setup logging
    args = ArgParser().parse()
    logger = logging.TrainingLogger(args)

    datasetPoses = datasets.Dataset(args.input, invertPose=args.invertPose, flipImage = args.flipImage)
    datasetScene = datasets.Dataset(args.input, invertPose=args.invertPose, flipImage = args.flipImage)
    dataloaderPoses = th.utils.data.DataLoader(datasetPoses, batch_size=1, shuffle=False, drop_last=False)
    dataloaderScene = th.utils.data.DataLoader(datasetScene, batch_size=args.sceneBatch, shuffle=True, drop_last=True)

    vertices, normals, colors = datasetScene.getPointset()
    vertices, normals, colors, stdDevs = datasets.setupTrainablePointset(vertices, normals, colors, args.stdDevBound, dtype=args.dtype, device=args.device)
    alphas = th.nn.Parameter(th.ones([1, vertices.shape[1], 1], device=args.device, dtype=args.dtype))
    with th.no_grad():
        normals.set_(normalize(normals))
    camPositions, camOrientations = datasetScene.getCameraPoses()
    camPoses = np.concatenate([camPositions, camOrientations], axis=-1)
    camPoses = th.nn.Parameter(th.from_numpy(camPoses).to(args.device, args.dtype))

    width, height, focallength, zmin, zmax = datasetScene.getCameraMetadata()
    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)

    ambientLight = th.tensor([0.0, 0.0, 1.0], dtype=args.dtype, device=args.device).view(1,1,3)
    lampDirections = th.tensor([0.0, 0.0, 1.0], dtype=args.dtype, device=args.device).view(1,1,3)
    lampIntesities = th.tensor([0.0, 0.0, 0.0], dtype=args.dtype, device=args.device).view(1,1,3)
    with th.no_grad():
        initialValues = th.rand_like(renderer.sphericalHarmonics.coefficients) * 0.2 - 0.1
        renderer.sphericalHarmonics.coefficients.set_(initialValues)
        renderer.sphericalHarmonics.coefficients[0,:,0] = math.pi*1.12837907
        renderer.forwardHarmonics.coefficients.set_(renderer.sphericalHarmonics.coefficients)

    #lossFn = losses.L1ImageLoss(reduction='sum')
    lossFn = losses.MaskedL1Loss(reduction='sum', threshold=0.01)
    poseLr = args.poseLr

    for step in range(args.steps):
        ###
        # Perform pose optimization
        ###
        posepath = os.path.join(logger.basepath, f"poses_step{step}")
        os.mkdir(posepath)
        loggerPose = logging.StepLogger(posepath, args)

        for batchIdx, (view, _, _, rgbImages, depthImages) in enumerate(dataloaderPoses):
            loggerPose.incrementEpoch()

            camPose = th.nn.Parameter(camPoses[view].view(1,7).to(args.device))
            optimizerPose = th.optim.Adam([camPose], lr=poseLr)
            schedulerPose = th.optim.lr_scheduler.StepLR(optimizerPose, step_size=args.poseDecaySteps, gamma=args.poseDecay)

            # Push tensors onto the GPU
            rgbImages = rgbImages.to(args.device)
            depthImages = depthImages.to(args.device)

            for epoch in range(args.poseEpochs):
                loggerPose.incrementBatch()
                optimizerPose.zero_grad()

                cameraPositions = camPose[:,:3].to(args.device)
                cameraOrientations = camPose[:, 3:].to(args.device)

                # Render image
                image, _, _ = renderer(cameraPositions, cameraOrientations, vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities, alphas)
                #loss, _ = lossFn(image, rgbImages)
                loss, _ = lossFn(rgbImages, image)

                # Backward pass
                loss.backward()
                optimizerPose.step()

                # Enforce constraints
                with th.no_grad():
                    camPose[:, 3:].set_(normalize(camPose[:, 3:]))

                print('Pose Step {} Frame {} Iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    step, view, epoch, batchIdx , len(datasetPoses),
                    100. * batchIdx / len(datasetPoses), loss.item()))

                # Write loss and gradient informations
                loggerPose.logLoss(loss, optimizerPose.param_groups[0]['lr'])
                loggerPose.logGradient(camPose.grad, "pose")

                # Store samples as an image
                if loggerPose.batch % 10 == 0:
                    loggerPose.logImages(image, [0,1], "optimized")
                    loggerPose.logImages(rgbImages, [0,1], "target")

                # Update scheduler after each epoch
                schedulerPose.step()
                print("Schedulre: Updated pose lr to", optimizerPose.param_groups[0]['lr'])

            # Rewrite scene parameter
            with th.no_grad():
                camPoses[view] = camPose[0]

            # Save scene parameters after each epoch
            loggerPose.logCameras(camPoses[...,:3], camPoses[..., 3:])
        poseLr *= args.poseDecay
        loggerPose.close()

        ###
        # Perform scene optimization
        ###
        optimizerScene = th.optim.Adam([colors, normals, alphas] + list(renderer.parameters()), lr=args.sceneLr) #+ list(renderer.parameters())
        schedulerScene = th.optim.lr_scheduler.StepLR(optimizerScene, step_size=args.sceneDecaySteps, gamma=args.sceneDecay)
        scenepath = os.path.join(logger.basepath, f"scene_step{step}")
        os.mkdir(scenepath)
        loggerScene = logging.StepLogger(scenepath, args)
        for epoch in range(args.sceneEpochs):
            loggerScene.incrementEpoch()

            for batchIdx, (views, _, _, rgbImages, depthImages) in enumerate(dataloaderScene):
                loggerScene.incrementBatch()

                outImages = th.empty_like(rgbImages)
                #outMasks = th.empty([args.batch, width, height, 1], device=args.device)
                batchLoss = 0

                optimizerScene.zero_grad()
                cameraPositions = camPoses[views, :3].to(args.device)
                cameraOrientations = camPoses[views, 3:].to(args.device)

                batch = [cameraPositions, cameraOrientations, rgbImages, depthImages]
                for idx, (cameraPosition, cameraOrientation, rgbImage, depthImage) in enumerate(zip(*batch)):

                    # Push tensors onto the GPU
                    rgbImage = rgbImage.to(args.device)
                    depthImage = depthImage.to(args.device)

                    # Render image
                    image, _, _ = renderer(cameraPosition[None].detach(), cameraOrientation[None].detach(), vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities, alphas)
                    #loss = lossFn(image, rgbImage[None])[0]/args.sceneBatch
                    loss = lossFn(rgbImage[None], image)[0]/args.sceneBatch

                    # Backward pass
                    loss.backward()

                    # Accumulate batch loss output image
                    with th.no_grad():
                        batchLoss += loss
                        outImages[idx] = image.detach()
                        #outMasks[idx] = mask.detach()
                optimizerScene.step()

                # Enforce constraints
                with th.no_grad():
                    normals.set_(normalize(normals))
                    colors.clamp_(0.0, 1.0)
                    stdDevs.set_(th.relu(stdDevs))
                    alphas.clamp_(0.0, 1.0)

                print('Scene Step {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    step, epoch, batchIdx , len(datasetScene),
                    100. * batchIdx / len(datasetScene), batchLoss.item()))

                # Write loss and gradient informations
                loggerScene.logLoss(batchLoss, optimizerScene.param_groups[0]['lr'])
                loggerScene.logGradient(colors.grad, "colors")
                loggerScene.logGradient(normals.grad, "normals")
                #loggerScene.logGradient(colors.grad, "coefficients")

                # Store samples as an image
                if loggerScene.batch % 5 == 0:
                    loggerScene.logViewIndices(views)
                    loggerScene.logImages(outImages, [0,1], "optimized")
                    #loggerScene.logImages(rgbImages, [0,1], "reference")

            # Update scheduler after each epoch
            schedulerScene.step()
            print("Schedulre: Updated scene lr to", optimizerScene.param_groups[0]['lr'])

            # Save scene parameters after each epoch
            loggerScene.logStdDevs(stdDevs)
            loggerScene.logLighting(renderer, lampDirections, lampIntesities)
        loggerScene.logPointset(vertices, normals, colors, alphas)

        ###
        # Perform scene optimization
        ###
        geometryParameter = []
        if args.optimizeAlpha is True:
            geometryParameter.append(alphas)
        if args.optimizePosition is True:
            geometryParameter.append(vertices)
        if not args.optimizeAlpha and not args.optimizePosition:
            args.geometryEpochs = 0
        else:
            optimizerGeometry = th.optim.Adam(geometryParameter, lr=args.geometryLr) 
            schedulerGeometry = th.optim.lr_scheduler.StepLR(optimizerGeometry, step_size=args.geometryDecaySteps, gamma=args.geometryDecay)
        geometrypath = os.path.join(logger.basepath, f"geometry_step{step}")
        os.mkdir(geometrypath)
        loggerGeometry = logging.StepLogger(geometrypath, args)
        #print(args.optimizeAlpha, args.optimizePosition, not args.optimizeAlpha and not args.optimizePosition, args.geometryEpochs)
        #input()
        for epoch in range(args.geometryEpochs):
            loggerGeometry.incrementEpoch()

            for batchIdx, (views, _, _, rgbImages, depthImages) in enumerate(dataloaderScene):
                loggerGeometry.incrementBatch()

                outImages = th.empty_like(rgbImages)
                #outMasks = th.empty([args.batch, width, height, 1], device=args.device)
                batchLoss = 0

                optimizerGeometry.zero_grad()
                cameraPositions = camPoses[views, :3].to(args.device)
                cameraOrientations = camPoses[views, 3:].to(args.device)

                batch = [cameraPositions, cameraOrientations, rgbImages, depthImages]
                for idx, (cameraPosition, cameraOrientation, rgbImage, depthImage) in enumerate(zip(*batch)):

                    # Push tensors onto the GPU
                    rgbImage = rgbImage.to(args.device)
                    depthImage = depthImage.to(args.device)

                    # Render image
                    image, _, _ = renderer(cameraPosition[None].detach(), cameraOrientation[None].detach(), vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities, alphas)
                    #loss = lossFn(image, rgbImage[None])[0]/args.sceneBatch
                    loss = lossFn(rgbImage[None], image)[0]/args.sceneBatch

                    # Backward pass
                    loss.backward()

                    # Accumulate batch loss output image
                    with th.no_grad():
                        batchLoss += loss
                        outImages[idx] = image.detach()
                        #outMasks[idx] = mask.detach()
                optimizerGeometry.step()

                # Enforce constraints
                with th.no_grad():
                    normals.set_(normalize(normals))
                    colors.clamp_(0.0, 1.0)
                    stdDevs.set_(th.relu(stdDevs))
                    alphas.clamp_(0.0, 1.0)

                print('Geometry Step {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    step, epoch, batchIdx , len(datasetScene),
                    100. * batchIdx / len(datasetScene), batchLoss.item()))

                # Write loss and gradient informations
                loggerGeometry.logLoss(batchLoss, optimizerGeometry.param_groups[0]['lr'])
                loggerGeometry.logGradient(vertices.grad, "vertices")
                loggerGeometry.logGradient(alphas.grad, "alphas")

                # Store samples as an image
                if loggerGeometry.batch % 5 == 0:
                    loggerGeometry.logViewIndices(views)
                    loggerGeometry.logImages(outImages, [0,1], "optimized")

            # Update scheduler after each epoch
            schedulerGeometry.step()
            print("Schedulre: Updated geometry lr to", optimizerGeometry.param_groups[0]['lr'])

            # Save scene parameters after each epoch
            loggerGeometry.logStdDevs(stdDevs)
            loggerGeometry.logLighting(renderer, lampDirections, lampIntesities)
        loggerGeometry.logPointset(vertices, normals, colors, alphas)

    
        if args.probabilistFilter is True:
            ###
            # Remove points with high error contribution
            ###
            with th.no_grad():
                accumulatedPointError = th.zeros([1, vertices.shape[1], 1], device=args.device, dtype=args.dtype)
                for batchIdx, (view, _, _, rgbImages, depthImages) in enumerate(dataloaderPoses):

                    cameraPositions = camPoses[view, :3].to(args.device)
                    cameraOrientations = camPoses[view, 3:].to(args.device)

                    # Push tensors onto the GPU
                    rgbImages = rgbImages.to(args.device)
                    depthImages = depthImages.to(args.device)

                    # Compute per point error contribution
                    image, indices, weights = renderer(cameraPositions, cameraOrientations, vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities)
                    mask = (rgbImages.sum(dim=-1, keepdims=True) > lossFn.threshold).to(rgbImages.dtype)
                    perPixelError = th.sum(mask*th.abs(image - rgbImages), dim=-1, keepdim=True)
                    perPointError = pixelgather.gather(indices, th.unsqueeze(perPixelError*weights, dim=-1), vertices.shape[1])
                    accumulatedPointError += perPointError

                # Store pointset before rejecting outier
                loggerGeometry.logPointset(vertices, normals, colors, accumulatedPointError)
                loggerGeometry.incrementEpoch()

                # Remove points based on probabilistic model
                filtered = accumulatedPointError[0, th.nonzero(accumulatedPointError[0,:,0] > args.errorMean), 0]
                std = th.sum(th.square(filtered-args.errorMean))/(filtered.shape[0]-1)
                b = 1/math.sqrt(2)*float(std)
                cutoffLaplace = -b*math.log(args.rejectionProb*b) + args.errorMean

                indices = th.nonzero(accumulatedPointError[0,:,0] < cutoffLaplace)[:,0]
                vertices = th.nn.Parameter(vertices[:, indices, :])
                normals = th.nn.Parameter(normals[:, indices, :])
                colors = th.nn.Parameter(colors[:, indices, :])
                alphas = th.nn.Parameter(alphas[:, indices, :])
                accumulatedPointError = accumulatedPointError[:, indices, :]

                # Store pointset with error contributions
                loggerGeometry.logPointset(vertices, normals, colors, accumulatedPointError)

    logger.close()
