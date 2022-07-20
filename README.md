## Unbiased Gradient Estimation for Differentiable Surface Splatting via Poisson Sampling

[[Project Page]](TODO) | [[Paper]](TODO) | [[Supplemental]](TODO)


This repository contains the code for the submission "Unbiased Gradient Estimation for Differentiable Surface Splatting via Poisson Sampling".

## Citation
Please cite the following publication:

<b><a style="font-weight:bold" href="TODO">Unbiased Gradient Estimation for Differentiable Surface Splatting via Poisson Sampling</a></b>
```
@inproceedings{mullerUnbiased2022,
   title = {Unbiased Gradient Estimation for Differentiable Surface Splatting via Poisson Sampling},
   author = {M{\"u}ller, Jan U. and Weinmann, Michael and Klein, Reinhard},
   year      = {2022},
}
```

## Installation

We use mini-conda to manage the required python packages. Use the following command to create and activate the necessary conda environment

    conda create --name fmmdr python=3.8.8 --file package-list.txt
    conda activate fmmdr`

Additional requirements need to be installed manually are: 

 * open3d - `python -m pip install --user open3d==0.13.0`
 * pytorch3d - Please follow the installation instruction provided in https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

 After installing the requiert dependicies, the custom cuda extensions need to be compiled and install by running

    python setup.py install

The implementation has been test on Ubunut 18.04 LTS with CUDA 10.2 and CUDA 11.5.

## Experiments

### Comparison - Image-based Shape Reconstruction

    python optimization_geometry.py <PathToDataset> --scale=0.6 --epochs=300 --decay=0.464 --decaySteps=60 --stdDevBound=[0.0045,0.0045] --lightMode=relative

    python optimization_geometry.py <PathToDataset> --scale=0.6 --epochs=300 --decay=0.464 --decaySteps=60 --stdDevBound=[0.005,0.005] --lightMode=relative

    python optimization_geometry.py <PathToDataset> --scale=1.5 --epochs=300 --decay=0.500 --decaySteps=100 --stdDevBound=[0.007,0.007] --lightMode=relative

    python optimization_geometry.py <PathToDataset> --scale=1.5 --epochs=300 --decay=0.464 --decaySteps=60 --stdDevBound=[0.009,0.009] --lightMode=relative

After the reconstruction terminated, the Chamfer and Hausdorff distance can be computed using

    python chamferHausdorffDistance.py <PathToReferencePointCloud> <PathToReconstructionPointCloud> --scale=<ScaleValue>

The scale argument is either 0.6 or 1.5 for the bunny/teapot and yoga1/yoga6 objects respectively. DSS (Yifan et al. 2019) scales the point clouds in their provided dataset, consequently this scaling needs to be taken into consideration when computing the Chamfer and Hausdorff distance.

### Application - Room-scale Scene Refinement

    python reconstruction_refinement.py <PathToDataset> --step=8 --poseLr=0.0005 --poseEpoch=20 --poseDecaySteps=10 --poseDecay=0.5 --sceneLr=0.005 --sceneBatch=16 --sceneEpoch=20 --sceneDecaySteps=15 --sceneDecay=0.5 --probabilistFilter=True --shadingMode=forwardHarmonics

### Application - Neural Rendering

Pre-training of point clouds

    python optimization_geometry.py <PathToDataset> --imgScale=256 --epochs=300 --lr=0.02 --decay=0.5 --decaySteps=60 --threshold=0.1 --stdDevBound=[0.005,0.005] --shadingMode=None

    python optimization_geometry.py <PathToDataset> --imgScale=256 --epochs=300 --lr=0.02 --decay=0.5 --decaySteps=60 --threshold=0.1 --stdDevBound=[0.005,0.005] --shadingMode=None

    python optimization_geometry.py <PathToDataset> --imgScale=256 --epochs=300 --lr=0.02 --decay=0.5 --decaySteps=60 --threshold=0.1 --stdDevBound=[0.005,0.005] --shadingMode=None

    python optimization_geometry.py <PathToDataset> --imgScale=256 --epochs=300 --lr=0.02 --decay=0.5 --decaySteps=60 --threshold=0.1 --stdDevBound=[0.005,0.005] --shadingMode=None

    python optimization_geometry.py <PathToDataset> --imgScale=256 --epochs=300 --lr=0.02 --decay=0.5 --decaySteps=60 --threshold=0.1 --stdDevBound=[0.005,0.005] --shadingMode=None

Joint point cloud and network optimization
    
    python neural_rendering.py <PathToPointCloud> <PathToImages> --priorRes=256 --imageRes=256 --epochs=2000 --batch=8 --lr=0.0002 --decay=0.5 --decayStep=1000 --threshold=0.0 --stdDevBound=[0.005,0.005] --shadingMode=None

    python neural_rendering.py <PathToPointCloud> <PathToImages>  --priorRes=256 --imageRes=256 --epochs=2000 --batch=8 --lr=0.0002 --decay=0.5 --decayStep=1000 --threshold=0.0 --stdDevBound=[0.005,0.005] --shadingMode=None

    python neural_rendering.py <PathToPointCloud> <PathToImages> --priorRes=256 --imageRes=256 --epochs=2000 --batch=8 --lr=0.0002 --decay=0.5 --decayStep=1000 --threshold=0.0 --stdDevBound=[0.005,0.005] --shadingMode=None

    python neural_rendering.py <PathToPointCloud> <PathToImages> --priorRes=256 --imageRes=256 --epochs=1000 --batch=8 --lr=0.0001 --decay=0.1 --decayStep=500 --threshold=0.0 --stdDevBound=[0.005,0.005] --shadingMode=None

Inference of Images after pre-training and joint optimzation 

    python neural_inference.py <PathToTrainedModel>

## Contact
Jan Müller - <a href="mailto:muellerj@cs.uni-bonn.de">muellerj@cs.uni-bonn.de</a>