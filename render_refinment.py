from typing import Tuple
import torch as th
import math
import argparse
import os
import shutil
import numpy as np
from plyfile import PlyData

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
        parser.add_argument("--invertPose", type=bool, default=False)
        parser.add_argument("--forward", type=str, default='z')
        parser.add_argument("--output", type=str, default='./Results')
        parser.add_argument("--batch", type=int, default=1)

        # Rendering parameters
        parser.add_argument("--smoothing", type=float, default=0.0)
        parser.add_argument("--stdDev", type=float, default=0.001)
        parser.add_argument("--numSamples", type=int, default=40)
        parser.add_argument("--precision", type=float, default=0.01)
        parser.add_argument("--shadingMode", type=str, default="forwardHarmonics")
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

if __name__ == "__main__":

    # Parse the arguments and setup logging
    args = ArgParser().parse()
    logger = logging.TrainingLogger(args)

    # Load and setup the dataset
    dataset = datasets.PoseDataset(args.input, invertPose=args.invertPose, pointset=args.pointset)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=False)
    camPositions, camOrientations = dataset.getCameraPoses()
    try:
        poses = np.load(os.path.join(args.input, "cameras.npz"))
        camPositions = poses['arr_0']
        camOrientations = poses['arr_1']
    except:
        print("No optimized pose found.")
    camPoses = np.concatenate([camPositions, camOrientations], axis=-1)
    camPoses = th.nn.Parameter(th.from_numpy(camPoses).to(args.device, args.dtype))

    vertices, normals, colors = dataset.getPointset()
    vertices, normals, colors, stdDevs = datasets.setupTrainablePointset(vertices, normals, colors, [args.stdDev, args.stdDev], dtype=th.float32, device=args.device)
    normals = normalize(normals)

    alpha = None
    #try:
    plydata = PlyData.read(os.path.join(dataset._root, args.pointset))
    alphaArray = np.array(plydata['vertex'].data['scalar_error'])
    print(alphaArray.shape, alphaArray.dtype)
    alpha = th.from_numpy(alphaArray).to(args.device, args.dtype).view(1, alphaArray.shape[0], 1)
    input()
    #except:
    #    input("No alpha loaded")
    #    pass

    width, height, focallength, zmin, zmax = dataset.getCameraMetadata()
    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)

    # Setup lighting parameters
    ambientLight = th.tensor([0.0, 0.0, 0.0], dtype=args.dtype, device=args.device).view(1,1,3)
    lampDirections = th.tensor([0.0, 0.0, 1.0], dtype=args.dtype, device=args.device).view(1,1,3)
    lampIntesities = th.tensor([0.0, 0.0, 0.0], dtype=args.dtype, device=args.device).view(1,1,3)
    with th.no_grad():
        try:
            shCoefficients = np.load(os.path.join(args.input, "forwardHarmonics.npy"))
            renderer.forwardHarmonics.coefficients.set_(th.from_numpy(shCoefficients).cuda())
        except:
            renderer.forwardHarmonics.coefficients.detach().zero_()
            renderer.forwardHarmonics.coefficients[0,:,0] = math.pi*1.12837907
            print("Using sh coefficient fallback")

    with th.no_grad():
        for batchIdx, (viewIndices, _, _) in enumerate(dataloader):
            logger.incrementBatch()

            # Push data to the correct device
            print(batchIdx, viewIndices)
            cameraPosition = camPoses[batchIdx,:3].to(args.device)
            cameraOrientation = camPoses[batchIdx, 3:].to(args.device)

            image, indices, _ = renderer(cameraPosition[None], cameraOrientation[None], vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities, alpha)

            print('Rendering: [{}/{} ({:.0f}%)]'.format(
                batchIdx * len(cameraPosition), len(dataloader.dataset),
                100. * batchIdx / len(dataloader)))

            # Store samples as an image
            logger.logImages(image, [0,1], "optimized")

            th.cuda.empty_cache()
            
        # Save scene parameters after each epoch
        logger.close()
