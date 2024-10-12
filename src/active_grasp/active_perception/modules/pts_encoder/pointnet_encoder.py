from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from modules.pts_encoder.abstract_pts_encoder import PointsEncoder


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

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# NOTE: removed BN
class PointNetEncoder(PointsEncoder):

    def __init__(self, global_feat=True, in_dim=3, out_dim=1024, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.out_dim = out_dim
        self.feature_transform = feature_transform
        self.stn = STNkd(k=in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, out_dim, 1)
        self.global_feat = global_feat
        if self.feature_transform:
            self.f_stn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.shape[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.f_stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        point_feat = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], 1)

    def encode_points(self, pts):
        pts = pts.transpose(2, 1)
        if not self.global_feat:
            pts_feature = self(pts).transpose(2, 1)
        else:
            pts_feature = self(pts)
        return pts_feature


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 2500, 3))

    pointnet_global = PointNetEncoder(global_feat=True)
    out = pointnet_global.encode_points(sim_data)
    print("global feat", out.size())

    pointnet = PointNetEncoder(global_feat=False)
    out = pointnet.encode_points(sim_data)
    print("point feat", out.size())
