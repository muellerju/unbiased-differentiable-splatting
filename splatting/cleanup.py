import os
import torch as th
import open3d as o3d
import numpy as np

import splatting.dataset as datasets
from cudaExtensions import pixelgather
from render_dataset import generateLights

@th.no_grad()
def cleanup(
    args, logger, renderer,
    vertices, normals, colors, stdDevs,
    ambientLight
    ):

    infile = os.path.join(logger.basepath, f"pointset_epoch{args.epochs}.ply")
    outfile = os.path.join(logger.basepath, "pointset.ply")
    pointset_pcd = o3d.io.read_point_cloud(infile)
    output_pcd, ind = pointset_pcd.remove_statistical_outlier(nb_neighbors=13, std_ratio=2.0)
    o3d.io.write_point_cloud(outfile, output_pcd)

    vertices = th.from_numpy(np.asarray(output_pcd.points)).view(1, -1, 3).float().to(args.device)
    normals = th.from_numpy(np.asarray(output_pcd.normals)).view(1, -1, 3).float().to(args.device)
    colors = th.from_numpy(np.asarray(output_pcd.colors)).view(1, -1, 3).float().to(args.device)

    dataset = datasets.PoseDataset(args.input, invertPose=args.invertPose)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

    visibility = th.zeros([1, vertices.shape[1], 1], dtype=args.dtype, device=args.device)
    for batchIdx, (viewIndices, cameraPosition, cameraOrientation) in enumerate(dataloader):
        
        cameraPosition = cameraPosition.to(args.device)
        cameraOrientation = cameraOrientation.to(args.device)

        if args.lightMode == "relative":
            lampDirections, lampIntesities = generateLights(cameraPosition, cameraOrientation, args.forward)

        # Compute visiblity
        image, indices, weights = renderer(cameraPosition, cameraOrientation, vertices, normals, stdDevs, colors, ambientLight, lampDirections, lampIntesities)
        vertexContributions = pixelgather.gather(indices, weights.unsqueeze(-1), vertices.shape[1])
        visibility += (vertexContributions > args.visibilityThreshold).to(args.dtype)

    # Remove non visible points
    indices = th.nonzero(visibility[0,:,0] > args.visibilityFrames).squeeze()
    vertices = vertices[0, indices, :].unsqueeze(0)
    normals = normals[0, indices, :].unsqueeze(0)
    colors = colors[0, indices, :].unsqueeze(0)

    logger.logPointset(vertices, normals, colors)
    return vertices, normals, colors