import torch
import warnings

# from pointnet2.utils.pointnet2_utils import furthest_point_sample as fps
# from pointnet2.utils.pointnet2_utils import gather_operation as gather
from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
from pointnet2_ops.pointnet2_utils import gather_operation as gather
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPSSampler(torch.nn.Module):
    def __init__(
        self,
        num_out_points,
        permute,
        input_shape="bcn",
        output_shape="bcn",
        learn_noise=False,
        pointwise_dist=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.permute = permute
        self.name = "fps"

        if pointwise_dist:
            self.std_learner = STDLearner(num_out_points=num_out_points)

        elif learn_noise:
            self.noise_std = nn.Parameter(torch.tensor([1e-2]), requires_grad=True)
            self.noise_mean = nn.Parameter(torch.tensor([1e-2]), requires_grad=True)
        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("FPS: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.permute:
            _, N, _ = x.shape
            x = x[:, torch.randperm(N), :]

        idx = fps(x, self.num_out_points)

        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1).contiguous()
        y = gather(x, idx)
        if self.output_shape == "bnc":
            y = y.permute(0, 2, 1).contiguous()

        return y

    def get_std(self, x):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        orig = x[:, :, :]
        x = orig[:, :3, :]
        bs = x.shape[0]

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")
        else:
            y_std, y_mean = self.std_learner(x)  # bcn

            if self.output_shape == "bnc":
                y_std = y_std.permute(0, 2, 1)
                y_mean = y_mean.permute(0, 2, 1)

            y_std = y_std.contiguous()
            y_mean = y_mean.contiguous()

        return y_std, y_mean


class STDLearner(nn.Module):
    def __init__(self, k=3, bottleneck_size=128, num_out_points=64):
        super(STDLearner, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4a = nn.Linear(256, 3 * num_out_points)  # standard deviation head
        self.fc4b = nn.Linear(256, 3 * num_out_points)  # mean head

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.num_out_points = num_out_points

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        ya = self.fc4a(y)
        yb = self.fc4b(y)

        ya = ya.view(-1, 3, self.num_out_points)  # Batch x 3 x NumOutPoints
        yb = yb.view(-1, 3, self.num_out_points)  # Batch x 3 x NumOutPoints

        return ya, yb


if __name__ == "__main__":
    x = FPSSampler(64, False, learn_noise=True)
    x = torch.randn((1, 3, 64))
