import torch as th
import numpy as np
import math
import argparse
import shutil
import os
import time
import datetime

import splatting.logging as logging
import splatting.losses as losses 

import splatting.render as render
import splatting.dataset as datasets
import splatting.regularizier as regularizier
from utils.math import normalize, dot
from cudaExtensions import pixelgather
from render_dataset import generateLights
from splatting.cleanup import cleanup

"""class ArgDebugger:
    def __init__(self) -> None:
        # Dataset parameter
        self.input = './Dataset/whiteTeapot'
        self.flipImage = False
        self.invertPose = True
        self.forward = '-z'
        self.output = './Results'

        # General trainings parameters
        self.epochs = 250
        self.batch = 16

        # Optimization parameters
        self.lr = 0.01 #0.005
        self.decay = 0.870551
        self.decaySteps = 50

        # Distribution regularization parameters
        self.densityReg = 0.1
        self.depthThreshold = 0.1

        # Occluded point regularization parameters
        self.occlusionStep = 0.1
        self.threshold = 1.0
        self.visibilityDecay = 0.0

        # Rendering parameters
        self.smoothing = 0.0
        self.stdDevBound = [0.01, 0.1]
        self.numSamples = 40
        self.precision = 0.01
        self.shadingMode = "forward"

        # Technical parameters
        self.dtype = th.float32
        self.device = 'cuda:0'
"""

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # Dataset parameter
        parser.add_argument("input", type=str)
        parser.add_argument("--flipImage", type=bool, default=False)
        parser.add_argument("--invertPose", type=bool, default=True)
        parser.add_argument("--forward", type=str, default='-z')
        parser.add_argument("--output", type=str, default='./Results')
        parser.add_argument("--imgScale", type=int, default=-1)
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

        # Cleanup parameters
        parser.add_argument("--visibilityThreshold", type=float, default=0.1)
        parser.add_argument("--visibilityFrames", type=int, default=10)

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
    # PyTorch settings
    #th.autograd.set_detect_anomaly(True)

    # Parse the arguments and setup logging
    args = ArgParser().parse()
    logger = logging.TrainingLogger(args)

    # Load and setup the dataset
    dtype = th.float32
    dataset = datasets.Dataset(args.input, invertPose=args.invertPose, flipImage = args.flipImage, scale=args.imgScale)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False)

    vertices, normals, colors = dataset.getPointset()
    vertices, normals, colors, stdDevs = datasets.setupTrainablePointset(args.scale*vertices, normals, colors, [0.001, 0.003], dtype=args.dtype, device=args.device)

    width, height, focallength, zmin, zmax = dataset.getCameraMetadata()
    if args.imgScale > -1:
        focallength *= (args.imgScale/width)
        width = args.imgScale
        height = args.imgScale
    
    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)

    ambientLight, lampDirections, lampIntesities = datasets.loadSceneLighting(args.input, args.dtype, args.device)
    lampDirections = -lampDirections

    optimizer = th.optim.Adam([vertices, normals, colors, stdDevs], lr=args.lr)
    #optimizer = th.optim.SGD([vertices], lr=args.lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=args.decaySteps, gamma=args.decay)

    lossFn = losses.L1ImageLoss(reduction='sum')
    #lossFn = losses.L2ImageLoss(reduction='sum')
    #lossFn = losses.HuberImageLoss(0.1, reduction='sum')
    #lossFn = losses.MaskedL1Loss(0.01, reduction='mean')
    #lossFn = losses.MaskedL2Loss(0.01, reduction='mean')
    #lossFn = losses.MaskedHuberLoss(0.01, 0.1, reduction='mean')
    #lossFn = th.nn.MSELoss()
    #lossFn = losses.SmoothLoss(41, 50, []).to(args.device)

    #baseLossFn = losses.HuberImageLoss(0.1, reduction='mean')
    #baseLossFn = th.nn.MSELoss()
    #lossFn = losses.MultiScalarLoss(baseLossFn, [32/63., 16/63. , 8/63. , 4/63., 2/63. , 1/63.])
    #lossFn = losses.MultiScalarLoss(baseLossFn, [6/21., 5/21., 4/21., 3/21., 2/21., 1/21.])
    #lossFn = losses.MultiScalarLoss([0.0, 0.0, 0.0, 0.0, 1/2., 1/2.])
    #lossFn = losses.SmoothMultiScalarLoss(15, 15.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(args.device)

    #regFn = regularizier.DensityRegularizer(0.2, reduction='mean')
    #regFn = regularizier.SurfaceRegularizer(0.1, reduction='mean')
    regFn = regularizier.NearestSamplesRegularizer(args.regSigma, args.regNearest)
    visibility = th.zeros((1,vertices.shape[1], 1), device=args.device)

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

            if args.lightMode == "relative":
                lampDirections, lampIntesities = generateLights(cameraPosition, cameraOrientation, args.forward)

            # Rendering forward pass
            optimizer.zero_grad()
            start = time.time()
            image, indices, weights = renderer(cameraPosition, cameraOrientation, vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities)
            elapsed = time.time() - start
            print("Elapsed forward", elapsed/args.batch, "s")
            imageLoss, bluredImages = lossFn(image, rgbImage)
            mask = None

            # Evaluate optional regulaization terms
            normalLoss, surfaceLoss = 0.0, 0.0
            if (args.regNormal) > 0 or (args.regSurface > 0):
                normalLoss, surfaceLoss = regFn(indices, weights, vertices, normals)
            loss = imageLoss + args.regNormal*normalLoss + args.regSurface*surfaceLoss

            # Perform backward steps
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            elapsed = time.time() - start
            print("Elapsed backward", elapsed/args.batch, "s")
            #input()
            optimizer.step()

            # Enforce constraints on shading model
            with th.no_grad():
                normals.set_(normalize(normals))
                stdDevs.clamp_(*args.stdDevBound)

            #assert th.isfinite(vertices).all()
            #assert th.isfinite(normals).all()
            #assert th.isfinite(colors).all()
            #assert th.isfinite(stdDevs).all()

            # Move occuluded points in normal directions
            """with th.no_grad():
                if mask is not None:
                    visible = (mask*weights > args.visibilityThreshold).to(vertices.dtype)
                else:
                    visible = (weights > args.visibilityThreshold).to(vertices.dtype)
                visible = pixelgather.gather(indices, visible.unsqueeze(-1), vertices.shape[1])
                visible = th.clamp(visible.sum(dim=0, keepdim=True), 0, 1)
                visibility = (1-args.visibilityDecay)*visibility + visible"""

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchIdx * len(rgbImage), len(dataloader.dataset),
                100. * batchIdx / len(dataloader), loss.item()))

            # Write loss and gradient informations
            logger.logLosses([imageLoss, normalLoss, surfaceLoss, loss], optimizer.param_groups[0]['lr'])
            logger.logGradient(vertices.grad, "vertices")
            logger.logGradient(normals.grad, "normals")
            logger.logGradient(colors.grad, "colors")

            # Store samples as an image
            #logger.logSampling(renderer.indices)
            #logger.logImages(bluredImages, [0,1], "smoothed")
            logger.logViewIndices(viewIndices)
            logger.logImages(rgbImage, [0,1], "target")
            logger.logImages(image, [0,1], "optimized")
            iteration += 1
            th.cuda.empty_cache()

        # Reposition points which no longer contribute to the image
        with th.no_grad():
            # Gather indices of non-contributing points
            nonContributingIndices = th.nonzero(colors.sum(dim=-1).squeeze() < args.threshold).squeeze()
            contributingIndices = th.nonzero(colors.sum(dim=-1).squeeze() >= args.threshold).squeeze()

            if nonContributingIndices.dim() > 0 and nonContributingIndices.shape[0] > 0:
                logger.logRepositioning(nonContributingIndices)

                # Select new position for occluded points
                #assert nonContributingIndices.shape[0] <= contributingIndices.shape[0]
                #randomIndices = th.randperm(nonContributingIndices.shape[0], device=args.device)
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
                colors[0, nonContributingIndices, :].fill_(1.0)

        # Move occuluded points in normal directions
        """with th.no_grad():
            # Classify points as visible and occluded based on their frequency in the epoch
            visibilityLikelihood = visibility.squeeze()/len(dataloader)
            visibleIndices = th.nonzero(visibilityLikelihood >= args.likelihoodThreshold).squeeze()
            occludedIndices = th.nonzero(visibilityLikelihood < args.likelihoodThreshold).squeeze()

            if occludedIndices.shape[0] > 0:
                # Select new position for occluded points
                randomIndices = th.randperm(occludedIndices.shape[0], device=args.device)
                visibleIndices = visibleIndices[randomIndices]

                # Compute a randomized offset position from seleced visible point
                offset = th.randn((occludedIndices.shape[0], 3), dtype=args.dtype, device=args.device)
                offset = vertices[0,visibleIndices,:] + args.occlusionStep*normalize(offset)

                # Project offset position into the plane of the selected visible points
                orthogonal = dot(offset - vertices[0,visibleIndices,:], normals[0, visibleIndices,:])
                newPosition = offset - orthogonal * normals[0, visibleIndices,:]

                # Replace position of the occluded points the new position
                vertices[0,occludedIndices,:] = vertices[0, visibleIndices,:]
                normals[0, occludedIndices,:] = normals[0, visibleIndices,:]
                visibility.fill_(0)"""

        # Update scheduler after each epoch
        scheduler.step()
        #lossFn.decaySmoothing()
        print("Scheduler: Updated shading lr to", optimizer.param_groups[0]['lr'])

        # Save scene parameters after each epoch
        logger.logPointset(vertices, normals, colors)
        logger.logStdDevs(stdDevs)
        logger.logLighting(renderer, lampDirections, lampIntesities)

    vertices, normals, colors = cleanup(args, logger, renderer, vertices, normals, colors, stdDevs, ambientLight)
    logger.logPointset(vertices, normals, colors)
    logger.close()

    # Perform a simple out-lier removale step and copy data to be able to render the dataset
    shutil.copy(os.path.join(args.input, "calibration.yml"), os.path.join(logger.basepath, "calibration.yml"))
    shutil.copy(os.path.join(args.input, "pose.dat"), os.path.join(logger.basepath, "pose.dat"))

    #infile = os.path.join(logger.basepath, f"pointset_epoch{args.epochs}.ply")
    #outfile = os.path.join(logger.basepath, "pointset.ply")
    #pointset_pcd = o3d.io.read_point_cloud(infile)
    #output_pcd, ind = pointset_pcd.remove_statistical_outlier(nb_neighbors=13, std_ratio=2.0)
    #o3d.io.write_point_cloud(outfile, output_pcd)

    #shutil.copy(os.path.join(logger.basepath, f"lights_epoch{args.epochs}.json"), os.path.join(logger.basepath, "lights.json"))
