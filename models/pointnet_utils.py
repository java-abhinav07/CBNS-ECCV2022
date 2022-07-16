import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        global_feat=True,
        feature_transform=False,
        channel=3,
        num_out_points=64,
        use_enc_stn=True,
        num_in_points=1024,
    ):
        super(PointNetEncoder, self).__init__()
        self.use_enc_stn = use_enc_stn
        self.num_in_points = num_in_points
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.num_in_points == 4096:
            self.conv4 = torch.nn.Conv1d(1024, 2048, 1)
            self.conv5 = torch.nn.Conv1d(2048, 4096, 1)
            self.bn4 = nn.BatchNorm1d(2048)
            self.bn5 = nn.BatchNorm1d(4096)
        elif self.num_in_points == 2048:
            self.conv4 = torch.nn.Conv1d(1024, 2048, 1)
            self.bn4 = nn.BatchNorm1d(2048)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        if self.use_enc_stn:
            B, D, N = x.size()
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.num_in_points == 4096:
            x = F.relu(self.bn4(self.conv4(F.relu(x))))
            x = self.bn5(self.conv5(x))
        elif self.num_in_points == 2048:
            x = F.relu(self.bn4(self.conv4(F.relu(x))))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.num_in_points)  # bs, num_in_points
        if self.use_enc_stn:
            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(-1, self.num_in_points, 1).repeat(1, 1, N)
                return torch.cat([x, pointfeat], 1), trans, trans_feat
        else:
            if self.global_feat:
                return x, None, trans_feat
            else:
                x = x.view(-1, self.num_in_points, 1).repeat(1, 1, N)
                return torch.cat([x, pointfeat], 1), None, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()

    # add epsilon for line cloud
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


def visualize_pointcloud(
    points, normals=None, out_file=None, show=False, combined=False
):
    r""" Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    """
    # Use numpy
    if combined:
        points, additional = points

    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2])
    if normals is not None:
        ax.quiver(
            points[:, 2],
            points[:, 0],
            points[:, 1],
            normals[:, 2],
            normals[:, 0],
            normals[:, 1],
            length=0.1,
            color="k",
        )

    if combined:
        ax.scatter(
            additional[:, :, 0],
            additional[:, :, 1],
            additional[:, :, 2],
            c="yellow",
            edgecolor="red",
            s=120,
            linewidths=2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(-180, 90)

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_data(data, out_file, data_type="pointcloud", combined=False):
    r""" Visualizes the data with regard to its type.
    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    """
    if data_type == "pointcloud":
        visualize_pointcloud(data, out_file=out_file, combined=combined)
    elif data_type is None or data_type == "idx":
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)
