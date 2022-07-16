import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

try:
    from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
except:
    print("in except..")
    from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(
        self, k=20, normal_channel=False, use_enc_stn=True, num_in_points=1024
    ):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3

        self.num_in_points = num_in_points
        self.feat = PointNetEncoder(
            global_feat=True,
            feature_transform=True,
            channel=channel,
            num_out_points=k,
            use_enc_stn=use_enc_stn,
            num_in_points=num_in_points,
        )

        self.fc1 = nn.Linear(self.num_in_points, self.num_in_points // 2)
        self.fc2 = nn.Linear(self.num_in_points // 2, self.num_in_points // 4)
        self.fc3 = nn.Linear(self.num_in_points // 4, k)
        self.bn1 = nn.BatchNorm1d(self.num_in_points // 2)
        self.bn2 = nn.BatchNorm1d(self.num_in_points // 4)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)  # B, D, N
        x, trans, trans_feat = self.feat(x)  # B, 1024, 1024
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        # softmax taken in forward
        self.loss = nn.NLLLoss()

    def forward(self, pred, target, trans_feat):
        loss = self.loss(F.log_softmax(pred, dim=1), target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
