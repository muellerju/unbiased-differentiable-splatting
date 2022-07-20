from typing import Tuple
import torch as th
import math
import argparse
import os
import shutil
import numpy as np


import splatting.logging as logging
import splatting.losses as losses 

import splatting.render as render
import splatting.dataset as datasets
import splatting.regularizier as regularizier
from utils.math import normalize
from splatting.extensions.utils import world2Camera, transformForward

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Dataset parameter
        parser.add_argument("input", type=str)
        parser.add_argument("pointset", type=str)
        parser.add_argument("--pose", type=str, default="pose.dat")
        parser.add_argument("--invertPose", type=bool, default=True)
        parser.add_argument("--forward", type=str, default='-z')
        parser.add_argument("--output", type=str, default='./Results')
        parser.add_argument("--batch", type=int, default=2)

        # Rendering parameters
        parser.add_argument("--smoothing", type=float, default=0.0)
        parser.add_argument("--stdDev", type=float, default=0.01)
        parser.add_argument("--numSamples", type=int, default=40)
        parser.add_argument("--precision", type=float, default=0.01)
        parser.add_argument("--shadingMode", type=str, default="forward")
        parser.add_argument("--lightMode", type=str, default="fixed")
        parser.add_argument("--resolution", type=int, default=-1)

        # Technical parameters
        parser.add_argument("--dtype", type=str, default="float")
        parser.add_argument("--device", type=str, default='cuda:0')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        if args.dtype == "float":
            args.dtype = th.float32
        else:
            raise ValueError("Unknown dtype")
        args.device = th.device(args.device)
        return args

def generateLights(camPositions : th.Tensor, camQuaternion : th.Tensor, forward : str): # -> Tuple(th.Tensor, th.Tensor):
        camTransformation = world2Camera(camPositions, camQuaternion)
        camTransformation = transformForward(camTransformation, forward).squeeze(1) # [bn, 4, 4]
        camRotation = camTransformation[...,:3,:3] # [bn,3,3]
        #camForward = camRotation[...,2,:] #[bn, 3]
        
        if forward == "-z":
            camForward = th.tensor([0.0, 0.0, 1.0], dtype=camPositions.dtype)
        elif forward == "z":
            camForward = th.tensor([0.0, 0.0, -1.0], dtype=camPositions.dtype)
        else:
            raise ValueError()
        camForward = camForward.view(1, 3)
        print(camRotation.shape, camForward.shape)


        # R around camera position
        rDir = normalize(camForward).cuda()
        rColor = th.tensor([0.9, 0, 0], dtype=camForward.dtype).expand_as(rDir).cuda()

        #bDir = normalize((rDir + th.rand_like(rDir)).cross(rDir)).cuda()
        bDir = th.tensor([0.0, -1.0, 0.0], dtype=camForward.dtype).view(1, 3).cuda()
        bColor = th.tensor([0, 0.9, 0], dtype=camForward.dtype).expand_as(bDir).cuda()

        #gDir = (bDir).cross(rDir).cuda()
        gDir = th.tensor([0.0, 1.0, 0.0], dtype=camForward.dtype).view(1, 3).cuda()
        gColor = th.tensor([0, 0.0, 0.9], dtype=camForward.dtype).expand_as(bDir).cuda()

        lightDirections = th.stack([rDir, bDir, gDir], dim=-2) #[bn,l,3]
        lightDirections = th.inverse(camRotation).unsqueeze(1) @ lightDirections.unsqueeze(-1) # [bn, 1, 3, 3] x [bn, l, 3, 1] -> [bn,l,3,1]
        lightDirections = lightDirections.squeeze(-1) # [bn,l, 3]

        lightColors = math.pi*th.stack([rColor, bColor, gColor], dim=-2) #[bnl, 3]

        return lightDirections, lightColors

if __name__ == "__main__":

    # Parse the arguments and setup logging
    '''parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input', type=str, help='Path to the input dataset')
    parser.add_argument('output', type=str, help='Path to a directory in which the results are storeds')
    parser.add_argument('smoothing', type=float, default=1.0, help='This constant scaled the gaussian smoothing')
    parser.add_argument('epochs', type=int, default=10, help='Number of trainings epochs')
    parser.add_argument('lr', type=float, default=0.01, help='Learning rate used during optimization')
    args = parser.parse_args()'''

    #args = ArgDebugger()
    args = ArgParser().parse()
    logger = logging.TrainingLogger(args)

    # Load and setup the dataset
    dataset = datasets.PoseDataset(args.input, invertPose=args.invertPose, pointset=args.pointset)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    vertices, normals, colors = dataset.getPointset()
    vertices, normals, colors, stdDevs = datasets.setupTrainablePointset(vertices, normals, colors, [args.stdDev, args.stdDev], dtype=th.float32, device=args.device)
    normals = normalize(normals)
    #colors = th.ones_like(colors)

    width, height, focallength, zmin, zmax = dataset.getCameraMetadata()
    if args.resolution > 0:
        with th.no_grad():
            stdDevs *= args.resolution/width
        focallength *= args.resolution/width
        width = args.resolution
        height = args.resolution
    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)

    # Setup lighting parameters
    print("Imported scene lighting")
    try:
        ambientLight, lampDirections, lampIntesities = datasets.loadSceneLighting(args.input, args.dtype, args.device)
    except:
        ambientLight = th.tensor([0.0, 0.0, 0.0], dtype=args.dtype, device=args.device).view(1,1,3)
        lampDirections = th.tensor([0.0, 0.0, 1.0], dtype=args.dtype, device=args.device).view(1,1,3)
        lampIntesities = th.tensor([0.0, 0.0, 0.0], dtype=args.dtype, device=args.device).view(1,1,3)
    with th.no_grad():
        try:
            shCoefficients = np.load(os.path.join(args.input, "shCoefficients.npy"))
            renderer.sphericalHarmonics.coefficients.set_(th.from_numpy(shCoefficients).cuda())
        except:
            print("Cannot loader spherical harmonics")
            renderer.sphericalHarmonics.coefficients.detach().zero_()
            renderer.sphericalHarmonics.coefficients[0,:,0] = math.pi*1.12837907

    # Copy scene lighting to output
    try:
        shutil.copy(os.path.join(args.input, "lights.json"), os.path.join(logger.basepath, "lights.json"))
    except:
        print("Cannot copy scene lighting.")

    with th.no_grad():
        for batchIdx, (viewIndices, cameraPosition, cameraOrientation) in enumerate(dataloader):
            logger.incrementBatch()

            # Push data to the correct device
            cameraPosition = cameraPosition.to(args.device)
            cameraOrientation = cameraOrientation.to(args.device)
            #rgbImage = rgbImage.to(args.device)
            #depthImage = depthImage.to(args.device)
            if args.lightMode == "relative":
                lampDirections, lampIntesities = generateLights(cameraPosition, cameraOrientation, args.forward)

            image, indices, _ = renderer(cameraPosition, cameraOrientation, vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities)
            print(cameraPosition.shape, cameraOrientation.shape, vertices.shape, normals.shape, colors.shape, indices.shape, image.shape)

            print('Rendering: [{}/{} ({:.0f}%)]'.format(
                batchIdx * len(cameraPosition), len(dataloader.dataset),
                100. * batchIdx / len(dataloader)))

            # Store samples as an image
            #logger.logImages(rgbImage, [0,1], "target")
            logger.logImages(image, [0,1], "optimized")

            # Store values to compare to gt
            with th.no_grad():
                camPoints, covariances = renderer.computeCovariance(cameraPosition, cameraOrientation, vertices, normals, colors, stdDevs)
                logger.logCovariance(covariances)
                logger.logPoissonSamples(indices)
                np.save(os.path.join(logger.poissonpath, f"camPoints_Epoch{logger.epoch}_Batch{logger.batch}.npy"), camPoints.cpu().numpy())
    
            # Compute and store the fmm weight approximation
            #fmmWeights = renderer.computeFMMWeights(cameraPosition, cameraOrientation, vertices, normals, stdDevs)
            #logger.logFMMWeights(fmmWeights)

            th.cuda.empty_cache()
            #print(th.cuda.memory_summary())

        # Save scene parameters after each epoch
        logger.close()
