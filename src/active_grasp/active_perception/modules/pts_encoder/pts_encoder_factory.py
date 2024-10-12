import sys
import os
path = os.path.abspath(__file__)
for i in range(3):
    path = os.path.dirname(path)
PROJECT_ROOT = path
sys.path.append(PROJECT_ROOT)

from modules.pts_encoder.abstract_pts_encoder import PointsEncoder
from modules.pts_encoder.pointnet_encoder import PointNetEncoder
from modules.pts_encoder.pointnet2_encoder import PointNet2Encoder
from modules.pts_encoder.pointnet3_encoder import PointNet3Encoder

class PointsEncoderFactory:
    @staticmethod
    def create(name, config) -> PointsEncoder:
        general_config = config["general"]
        pts_encoder_config = config["pts_encoder"][name]
        if name == "pointnet":
            return PointNetEncoder(
                in_dim=general_config["pts_channels"],
                out_dim=general_config["feature_dim"],
                global_feat=not general_config["per_point_feature"]
            )
        elif name == "pointnet++":
            return PointNet2Encoder(
                input_channels=general_config["pts_channels"] - 3,
                params_name=pts_encoder_config["params_name"]
            )
        elif name == "pointnet++rgb":
            return PointNet3Encoder(
                input_channels=general_config["pts_channels"] - 3,
                params_name=pts_encoder_config["params_name"],
                target_layer=pts_encoder_config["target_layer"],
                rgb_feat_dim=pts_encoder_config["rgb_feat_dim"]
            )
        else:
            raise ValueError(f"Unknown encoder name: {name}")


''' ------------ Debug ------------ '''
if __name__ == "__main__":
    from configs.config import ConfigManager
    import torch

    pts = torch.rand(32, 1200, 3)  # BxNxC
    ConfigManager.load_config_with('configs/local_train_config.yaml')
    ConfigManager.print_config()
    pts_encoder = PointsEncoderFactory.create(name="pointnet++", config=ConfigManager.get("modules"))
    print(pts_encoder)
    pts = pts.to("cuda")
    pts_encoder = pts_encoder.to("cuda")

    pts_feat = pts_encoder.encode_points(pts)

    print(pts_feat.shape)
