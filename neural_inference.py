import torch as th
import numpy as np
import math
import argparse
import shutil
import os

import splatting.logging as logging
import splatting.losses as losses 
import splatting.render as render
import splatting.dataset as datasets
import splatting.regularizier as regularizier

from utils.math import normalize, dot
from cudaExtensions import pixelgather

from render_dataset import ArgParser
from neural_rendering import UNet, DropoutUNet

def load_ckp(checkpoint_fpath, model, optimizer):
    # load check point
    checkpoint = th.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    return checkpoint['epoch'], model, optimizer

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Dataset parameter
        parser.add_argument("input", type=str)
        parser.add_argument("model", type=str)
        parser.add_argument("--pose", type=str, default="testpose.dat")
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

        # Model paramters
        parser.add_argument("--genDepth", type=int, default=4)
        parser.add_argument("--genInDim", type=int, default=16)
        parser.add_argument("--genFeatures", type=int, default=64)
        parser.add_argument("--discDepth", type=int, default=3)
        parser.add_argument("--discFeatures", type=int, default=64)

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
    dataset = datasets.PoseDataset(args.input, invertPose=args.invertPose, pose=args.pose)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    vertices, normals, _ = datasets.loadPointset(os.path.join(args.model, "pointset.ply"), np.float32)
    features = np.load(os.path.join(args.model, "features.ply"))[0]
    vertices, normals, features, stdDevs = datasets.setupTrainablePointset(vertices, normals, features, [args.stdDev, args.stdDev], dtype=th.float32, device=args.device)

    width, height, focallength, zmin, zmax = dataset.getCameraMetadata()
    if args.resolution > -1:
        focallength *= (args.resolution/width)
        width = args.resolution
        height = args.resolution

    renderer = render.Splatting(args, logger, width, height, focallength, zmin, zmax, smoothing=args.smoothing)
    renderer.to(args.device)

    ambientLight, lampDirections, lampIntesities = datasets.loadSceneLighting(args.input, args.dtype, args.device)

    # Load model state
    network = UNet(args.genDepth, args.genInDim, args.genFeatures).to(args.device)
    #network = DropoutUNet(args.genDepth, args.genInDim, args.genFeatures).to(args.device)

    load_ckp(os.path.join(args.model, "neural_generator.pt"), network, None)
    network.eval()
    
    with th.no_grad():
        for batchIdx, (viewIndices, cameraPosition, cameraOrientation) in enumerate(dataloader):
            logger.incrementBatch()

            # Push data to the correct device
            cameraPosition = cameraPosition.to(args.device)
            cameraOrientation = cameraOrientation.to(args.device)

            print(features.shape)
            priorImage, _, _ = renderer(cameraPosition, cameraOrientation, vertices, normals, stdDevs, features, ambientLight, lampDirections, lampIntesities)
            image = network(priorImage)

            print('Rendering: [{}/{} ({:.0f}%)]'.format(
                batchIdx * len(cameraPosition), len(dataloader.dataset),
                100. * batchIdx / len(dataloader)))

            # Store samples as an image
            #logger.logImages(priorImage, [0,1], "prior")
            logger.logImages(image, [0,1], "neural")

            th.cuda.empty_cache()

        # Save scene parameters after each epoch
        logger.close()
