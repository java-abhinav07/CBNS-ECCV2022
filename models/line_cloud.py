import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

try:
    import src.pctransforms as d_utils
except:
    PACKAGE_PARENT = ".."
    SCRIPT_DIR = os.path.dirname(
        os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
    )
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    import src.pctransforms as d_utils

tf = transforms.Compose([d_utils.OnUnitCube()])


def get_line_cloud(x):
    """
    L: (x - x1)/l = (y - y1)/m = (z - z1)/n
    l = x1 - x2
    m = y1- y2
    n = z1- z2
    x: B, N, D (B, 64, 3)
    """
    while True:
        # random direction
        x2 = torch.randn_like(x).to(x.device)
        DR = x - x2
        l, m, n = DR[:, :, :1], DR[:, :, 1:2], DR[:, :, 2:3]

        # sample random points from line
        x_sampled = torch.randn_like(l).to(x.device)
        c = (x_sampled - x[:, :, :1]) / l
        y_sampled = (c * m) + x[:, :, 1:2]
        z_sampled = (c * n) + x[:, :, 2:3]

        new_points = torch.cat([x_sampled, y_sampled, z_sampled], dim=-1)

        # normalize in unit circle
        new_points = tf(new_points)

        assert new_points.shape == x.shape

        if new_points.sum() > 0:
            break

    return new_points
