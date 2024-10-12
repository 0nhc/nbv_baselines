import torch
import torch.nn as nn
from modules.module_lib.pointnet2_utils.pointnet2.pointnet2_modules import PointnetSAModuleMSG
from modules.pts_encoder.abstract_pts_encoder import PointsEncoder

ClsMSG_CFG_Dense = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[32, 64], [16, 32], [8, 16], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG_Light = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]],
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG_Lighter = {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01], [0.02], [0.04], [0.08], [None]],
    'NSAMPLE': [[64], [32], [16], [8], [None]],
    'MLPS': [[[32, 32, 64]],
             [[64, 64, 128]],
             [[128, 196, 256]],
             [[256, 256, 512]],
             [[512, 512, 1024]]],
    'DP_RATIO': 0.5,
}


def select_params(name):
    if name == 'light':
        return ClsMSG_CFG_Light
    elif name == 'lighter':
        return ClsMSG_CFG_Lighter
    elif name == 'dense':
        return ClsMSG_CFG_Dense
    else:
        raise NotImplementedError


def break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )

    return xyz, features


class PointNet3Encoder(PointsEncoder):
    def encode_points(self, pts, rgb_feat):
        return self.forward(pts,rgb_feat)

    def __init__(self, input_channels=6, params_name="light",target_layer=2, rgb_feat_dim=384):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        self.target_layer = target_layer
        selected_params = select_params(params_name)
        for k in range(selected_params['NPOINTS'].__len__()):
            mlps = selected_params['MLPS'][k].copy()
            channel_out = 0
            if k==target_layer:
                channel_in += rgb_feat_dim
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=selected_params['NPOINTS'][k],
                    radii=selected_params['RADIUS'][k],
                    nsamples=selected_params['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            channel_in = channel_out

    def forward(self, point_cloud: torch.cuda.FloatTensor, rgb_feat):
        xyz, features = break_up_pc(point_cloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i==self.target_layer:
                rgb_feat = torch.mean(rgb_feat, dim=1)
                rgb_feat = rgb_feat.unsqueeze(-1).repeat(1,1,l_xyz[i].shape[1])
                l_features[-1] = torch.cat([l_features[-1], rgb_feat], dim=1)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_features[-1].squeeze(-1)


if __name__ == '__main__':
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    net = PointNet3Encoder(0).cuda()
    pts = torch.randn(2, 1024, 3).cuda()
    rgb_feat = torch.randn(2, 384).cuda()
    print(torch.mean(pts, dim=1))
    pre = net.encode_points(pts,rgb_feat)
    print(pre.shape)
