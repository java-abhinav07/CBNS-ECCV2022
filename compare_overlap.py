import os
import numpy as np
from tqdm import tqdm


def get_iou(pc1, pc2, pc1_points, top=100):
    """
        We define Mean IOU as: intersection / union of points 
        in the two corresponding top n point clouds.
        
    """

    # get top 100
    pc1, pc2 = pc1[-top:], pc2[-top:]

    # get nearby points to pc1
    new_indices = get_nearest(pc1_points.reshape(-1, 3), pc1)

    pc1 = list(pc1) + new_indices

    pc1, pc2 = set(pc1), set(pc2)

    # miou
    miou = len(pc1.intersection(pc2)) / top

    return miou


def get_nearest(points, indices, threshold=0.05):
    curr_points = points[indices, :]
    indices = []
    for i, point1 in enumerate(curr_points):
        for j, point2 in enumerate(points):
            if get_distance(point1, point2) < threshold:
                indices.append(j)

    #     print(len(indices))
    return indices


def get_distance(a, b):
    d = np.linalg.norm(a - b)
    return d


if __name__ == "__main__":
    ROOT1 = "modelnet/log_64/baseline/all/SaliencySnapshots/indices/"
    ROOT2 = "modelnet/log_64/baseline/living/SaliencySnapshots/indices/"
    ROOT3 = "modelnet/log_64/baseline/all/SaliencySnapshots/Points/"

    top_n = {}

    A = os.listdir(ROOT1)
    B = os.listdir(ROOT2)
    A.sort()
    B.sort()

    for n in tqdm([100, 50, 64]):
        iou_list = []

        for pc1, pc2 in zip(A, B):
            assert pc1 == pc2

            pc1_points = np.load(os.path.join(ROOT3, pc1), allow_pickle=True)
            pc1 = np.load(os.path.join(ROOT1, pc1), allow_pickle=True)
            pc2 = np.load(os.path.join(ROOT2, pc2), allow_pickle=True)

            miou = get_iou(pc1, pc2, pc1_points, top=n)
            iou_list.append(miou)

        miou_ = np.mean(iou_list)
        print()
        print(miou_)
        top_n[n] = miou_

    print("ModelNet")
    print(top_n)

    print("--" * 32)

    ROOT1 = "log_64/exp/plot_pointnet/indices/"
    ROOT2 = "log_64/gender/plot_pointnet/indices/"
    ROOT3 = "log_64/exp/plot_pointnet/Points/"

    top_n = {}

    A = os.listdir(ROOT1)
    B = os.listdir(ROOT2)
    A.sort()
    B.sort()

    for n in tqdm([100, 50, 64]):
        iou_list = []

        for pc1, pc2 in zip(A, B):
            assert pc1 == pc2

            pc1_points = np.load(os.path.join(ROOT3, pc1), allow_pickle=True)
            pc1 = np.load(os.path.join(ROOT1, pc1), allow_pickle=True)
            pc2 = np.load(os.path.join(ROOT2, pc2), allow_pickle=True)

            miou = get_iou(pc1, pc2, pc1_points, top=n)
            iou_list.append(miou)

        miou_ = np.mean(iou_list)
        print()
        print(miou_)
        top_n[n] = miou_

    print("FaceScape")
    print(top_n)

    print("--" * 32)
