import torch as th
import numpy as np
from PIL import Image
import os
import datetime
import matplotlib.pyplot as plt
import csv
import json
from typing import *
from plyfile import PlyData, PlyElement
from .dataset import storePointset

def _createDir(path : str):
    if not os.path.isdir(path):
        os.makedirs(path)

def _storeParameter(path, args):
    with open(os.path.join(path, "parameter.txt"), mode='w') as file:
        argStart = "----------Parameters----------\n"
        file.write(argStart)
        print(argStart, end='')
        for arg in vars(args):
            argString = "{} : {}\n".format(arg, getattr(args, arg))
            file.write(argString)
            print(argString, end='')
        argEnd = "------------------------------\n"
        file.write(argEnd)
        print(argEnd, end='')

class StepLogger:

    def __init__(self, basepath, args) -> None:
        
        self.basepath = basepath
        self.imagepath = os.path.join(self.basepath, "images")
        self.covariancepath = os.path.join(self.basepath, "covariance")
        self.poissonpath = os.path.join(self.basepath, "poisson")
        _createDir(self.basepath)
        _createDir(self.imagepath)
        _createDir(self.covariancepath)
        _createDir(self.poissonpath)
        _storeParameter(self.basepath, args)

        self.lossfp = open(os.path.join(self.basepath, 'loss.csv'), 'w', newline='')
        self.gradfp = open(os.path.join(self.basepath, 'grad.csv'), 'w', newline='')
        self.losslog = csv.writer(self.lossfp, delimiter=';', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        self.gradlog = csv.writer(self.gradfp, delimiter=';', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        self.losslog.writerow(["epoch", "batch", "target", "loss", "lr"])
        self.gradlog.writerow(["epoch", "batch", "name", "avg. value", "min. value", "max. value"])

        self.samplingLog = open(os.path.join(self.basepath, 'sampling.txt'), 'w')
        self.repositionfp = open(os.path.join(self.basepath, 'reposition.csv'), 'w', newline='')
        self.repositionlog = csv.writer(self.repositionfp, delimiter=';', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        self.repositionlog.writerow(['epoch', 'value'])
        self.shadingMode = args.shadingMode
        self.lightMode = args.lightMode
        
        self.epoch = 0
        self.batch = 0

    def close(self):
        self.lossfp.close()
        self.gradfp.close()
        self.samplingLog.close()
        self.repositionfp.close()

    def flush(self):
        self.lossfp.flush()
        self.gradfp.flush()
        self.samplingLog.flush()
        self.repositionfp.flush()

    def incrementEpoch(self):
        self.epoch += 1
        self.batch = 0
        self.flush()

    def incrementBatch(self):
        self.batch += 1

    @th.no_grad()
    def logRepositioning(self, indices : th.Tensor ):
        self.repositionlog.writerow([self.epoch, len(indices)])

    @th.no_grad()
    def logViewIndices(self, indices : th.Tensor):
        for idx, index in enumerate(indices):
            self.samplingLog.write(f"{self.epoch}; {index.item()}; epoch{self.epoch}_batch{self.batch}_element{idx}_\n")

    @th.no_grad()
    def logPoissonSamples(self, tensor : th.Tensor):
        indices = tensor.cpu().numpy()
        filename = os.path.join(self.poissonpath, "fmmSampled_epoch{}_batch{}.npy".format(self.epoch, self.batch))
        np.save(filename, indices)

    @th.no_grad()
    def logFMMWeights(self, tensor : th.Tensor):
        fmmWeights = tensor.cpu().numpy()
        filename = os.path.join(self.poissonpath, "fmmWeights_epoch{}_batch{}.npy".format(self.epoch, self.batch))
        np.save(filename, fmmWeights)

    @th.no_grad()
    def logImages(self, tensor : th.Tensor, interval : Tuple, suffix : str):
        bn = tensor.shape[0]
        images = tensor.cpu().transpose(1,2).numpy()
        images = np.clip(images, interval[0], interval[1])
        images = 255.0*(images - interval[0])/(interval[1] - interval[0])
        images = images.astype(np.uint8)
        for i in range(bn):
            if images.shape[-1] == 3:
                im = Image.fromarray(images[i,...], mode='RGB')
            else:
                im = Image.fromarray(images[i,...,0], mode='L')
            filename = "epoch{}_batch{}_element{}_{}.png".format(self.epoch, self.batch, i, suffix)
            im.save(os.path.join(self.imagepath, filename))

    @th.no_grad()
    def logNormals(self, tensor : th.Tensor, suffix : str):
        bn = tensor.shape[0]
        images = tensor.cpu().transpose(1,2).numpy()
        for i in range(bn):
            image = images[i,...]
            image[...,2] = -image[...,2]
            image = (image + 1)/2
            image = (255.0*image).astype(np.uint8)
            im = Image.fromarray(image, mode='RGB')
            filename = "epoch{}_batch{}_element{}_{}.png".format(self.epoch, self.batch, i, suffix)
            im.save(os.path.join(self.imagepath, filename))

    @th.no_grad()
    def logSampling(self, indices):
        bn, w, h, m = indices.shape

        validIndices = indices >= 0
        sampleImage = validIndices.float().sum(dim=-1)/m
    
        for i in range(bn):
            fig = plt.figure(3)
            fig.clear(True)
            ax = fig.add_subplot(111)
            im = ax.imshow(sampleImage[i,...].transpose(0,1).numpy(), vmin=0.0, vmax=1.0)
            cbar = ax.figure.colorbar(im, ax=ax)
            filename = "epoch{}_batch{}_element{}_sampling.png".format(self.epoch, self.batch, i)
            fig.savefig(os.path.join(self.imagepath, filename), dpi=300)

    @th.no_grad()
    def logLoss(self, loss : th.Tensor, lr : float, target : str = "scene"):
        value = str(loss.cpu().item()).replace('.', ',')
        self.losslog.writerow([self.epoch, self.batch, target, value, lr])

    @th.no_grad()
    def logLosses(self, losses, lr : float):
        values = [str(float(loss)).replace('.', ',') for loss in losses]
        self.losslog.writerow([self.epoch, self.batch, lr]+values)

    @th.no_grad()
    def logGradient(self, grad : th.Tensor, prefix : str):
        self.gradlog.writerow([self.epoch, self.batch, prefix, grad.cpu().abs().mean().item(), grad.cpu().min().item(), grad.cpu().max().item()])

    @th.no_grad()
    def logPointset(self, vertices : th.Tensor, normals : th.Tensor, colors : th.Tensor, error = None):
        filename = os.path.join(self.basepath, "pointset_epoch{}.ply".format(self.epoch))
        if error is not None:
            verts = np.concatenate((vertices.cpu().numpy(), normals.cpu().numpy(), (255*colors.clamp(0.0, 1.0)).cpu().numpy(), error.cpu().numpy()), axis=-1)[0]
            verts = [tuple(row) for row in verts]
            elV = PlyElement.describe(np.array(verts, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('error', 'f4')]), 'vertex')
        else:
            verts = np.concatenate((vertices.cpu().numpy(), normals.cpu().numpy(), (255*colors.clamp(0.0, 1.0)).cpu().numpy()), axis=-1)[0]
            verts = [tuple(row) for row in verts]
            elV = PlyElement.describe(np.array(verts, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        filename = os.path.join(self.basepath, "pointset_epoch{}.ply".format(self.epoch))
        PlyData([elV], text=False).write(filename)

        #storePointset(filename, vertices.cpu(), normals.cpu(), colors.cpu())

    @th.no_grad()
    def logStdDevs(self, tensor : th.Tensor):
        stdDevs = tensor.cpu().numpy()
        filename = os.path.join(self.basepath, "stdDevs_epoch{}.npy".format(self.epoch))
        np.save(filename, stdDevs)

    @th.no_grad()
    def logLighting(self, renderer, directions : th.Tensor, intensities : th.Tensor):

        if self.shadingMode == "sphericalHarmonics":
            coefficients = renderer.sphericalHarmonics.coefficients.cpu().squeeze().numpy()

            filename = 'shCoefficients_epoch{}.npy'.format(self.epoch)
            np.save(os.path.join(self.basepath, filename), coefficients)
        elif self.shadingMode == "forwardHarmonics":
            coefficients = renderer.forwardHarmonics.coefficients.cpu().squeeze().numpy()

            filename = 'forwardHarmonics_epoch{}.npy'.format(self.epoch)
            np.save(os.path.join(self.basepath, filename), coefficients)
        elif self.lightMode == "fixed":
            directions = directions.cpu().squeeze(0).numpy()
            intensities = intensities.cpu().squeeze(0).numpy()

            filename = 'lights_epoch{}.npz'.format(self.epoch)
            np.savez(os.path.join(self.basepath, filename), directions, intensities)

            json_dict = []
            for i, (direction, intensity) in enumerate(zip(directions, intensities)):
                color = 1/3.14*intensity
                entry = {
                    'Direction' : [float(direction[0]), float(direction[1]), float(direction[2])],
                    'Color' : [float(color[0]), float(color[1]), float(color[2])]
                }
                json_dict.append(entry)
            with open(os.path.join(self.basepath, f'lights_epoch{self.epoch}.json'), 'w') as file:
                json_object = json.dumps(json_dict, indent = 4)
                file.write(json_object)

    @th.no_grad()
    def logCovariance(self, tensor : th.Tensor):
        covariances = tensor.cpu().numpy()
        filename = os.path.join(self.covariancepath, "epoch{}_batch{}.npy".format(self.epoch, self.batch))
        np.save(filename, covariances)

    @th.no_grad()
    def logCameras(self, camPositions : th.Tensor, camOrientations : th.Tensor) -> None:
        if camPositions.dim() == 3:
            camMatrix = camPositions.cpu().numpy()
            filename = os.path.join(self.basepath, "cameras_epoch{}.npz".format(self.epoch))
            np.save(filename, camMatrix)
        else:
            camPositions = camPositions.cpu().numpy()
            camOrientations = camOrientations.cpu().numpy()
            filename = os.path.join(self.basepath, "cameras_epoch{}.npz".format(self.epoch))
            np.savez(filename, camPositions, camOrientations)
        
    @th.no_grad()
    def logNeuralState(self, model, optimizer, name) -> None:
        state = {
            'epoch': self.epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        fname = os.path.join(self.basepath, f"neural_{name}_{self.epoch}.pt")
        th.save(state, fname)

    @th.no_grad()
    def logFeatures(self, features) -> None:
        features = features.cpu().numpy()
        filename = os.path.join(self.basepath, f"features_epoch{self.epoch}.npy")
        np.save(filename, features)

class TrainingLogger(StepLogger):

    def __init__(self, args):
        # Create ouput folder and store parameters
        now = datetime.datetime.now()
        startDate = now.strftime("%b-%d-%Y-%H-%M")
        print("Logging started", startDate)
        basename = os.path.basename(os.path.normpath(args.input))
        basepath = os.path.join(args.output, basename+"_"+startDate)
        super().__init__(basepath, args)
