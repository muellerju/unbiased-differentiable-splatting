import numpy as np
import open3d as o3d
import argparse

class ArgParser:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Dataset parameter
        parser.add_argument("reference", type=str)
        parser.add_argument("reconstruction", type=str)
        parser.add_argument("--scale", type=float, default=1)

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def computeClostestDistances(A, B):
    pcd_tree = o3d.geometry.KDTreeFlann(B)
    distances = []
    for point in A.points:
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, 2)
        distances.append(dist[0])
    return distances

if __name__ == "__main__":

    args = ArgParser().parse()

    # Load both pointsets and scale the reference pointset
    A = o3d.io.read_point_cloud(args.reference)
    B = o3d.io.read_point_cloud(args.reconstruction)
    for idx, point in enumerate(A.points):
        A.points[idx] = args.scale*point

    # Compute nearest distances
    distancesAB = computeClostestDistances(A, B)
    distancesBA = computeClostestDistances(B, A)    

    # Compute Hausdorff distance
    print("Hausdorff distance", max(np.max(distancesAB), np.max(distancesBA)))

    # Compute chamfer distance
    print("Chamfer distance", max(np.mean(distancesAB), np.mean(distancesBA)))
