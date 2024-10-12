import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import torch
from torch import nn
import inspect

from configs.config import ConfigManager

from modules.pts_encoder.pts_encoder_factory import PointsEncoderFactory
from modules.view_finder.view_finder_factory import ViewFinderFactory
from modules.module_lib.fusion_layer import FeatureFusion
from modules.rgb_encoder.rgb_encoder_factory import RGBEncoderFactory


class Pipeline(nn.Module):
    TRAIN_MODE: str = "train"
    TEST_MODE: str = "test"

    def __init__(self, config_path):
        super(Pipeline, self).__init__()
        ConfigManager.load_config_with(config_path)
        pipeline_config = ConfigManager.get("settings", "pipeline")
        self.modules_config = ConfigManager.get("modules")
        self.device = ConfigManager.get("settings", "general", "device")
        self.rgb_feat_cache = ConfigManager.get("datasets", "general", "rgb_feat_cache")
        self.pts_encoder = PointsEncoderFactory.create(pipeline_config["pts_encoder"], self.modules_config)
        self.view_finder = ViewFinderFactory.create(pipeline_config["view_finder"], self.modules_config)
        self.has_rgb_encoder = "rgb_encoder" in pipeline_config
        if self.has_rgb_encoder and not self.rgb_feat_cache:
            self.rgb_encoder = RGBEncoderFactory.create(pipeline_config["rgb_encoder"], self.modules_config)
        self.eps = 1e-5
        self.fusion_layer = FeatureFusion(rgb_dim=384, pts_dim=1024,output_dim=1024)

        self.to(self.device)

    def forward(self, data, mode):
        if mode == self.TRAIN_MODE:
            return self.forward_gradient(data)
        elif mode == self.TEST_MODE:
            return self.forward_view(data)
        raise ValueError("Unknown mode: {}".format(self.mode))

    def forward_gradient(self, data):
        target_pts = data["target_pts"]
        scene_pts = data["scene_pts"]
        gt_delta_rot_6d = data["delta_rot_6d"]
        
        if hasattr(self,"rgb_encoder"): 
            if "rgb" in data:
                rgb_feat = self.rgb_encoder.encode_rgb(data["rgb"])
            else:
                rgb_feat = data["rgb_feat"]
            if "rgb_feat" not in inspect.signature(self.pts_encoder.encode_points).parameters:
                target_feat = self.pts_encoder.encode_points(target_pts)
                scene_feat = self.pts_encoder.encode_points(scene_pts)
                target_feat = self.fusion_layer(rgb_feat, target_feat)
                scene_feat = self.fusion_layer(rgb_feat, scene_feat)
            else:
                target_feat = self.pts_encoder.encode_points(target_pts, rgb_feat)
                scene_feat = self.pts_encoder.encode_points(scene_pts, rgb_feat)
        else:
            target_feat = self.pts_encoder.encode_points(target_pts)
            scene_feat = self.pts_encoder.encode_points(scene_pts)
        ''' get std '''
        bs = target_pts.shape[0]
        random_t = torch.rand(bs, device=self.device) * (1. - self.eps) + self.eps
        random_t = random_t.unsqueeze(-1)
        mu, std = self.view_finder.marginal_prob(gt_delta_rot_6d, random_t)
        std = std.view(-1, 1)

        ''' perturb data and get estimated score '''
        z = torch.randn_like(gt_delta_rot_6d)
        perturbed_x = mu + z * std
        input_data = {
            "sampled_pose": perturbed_x,
            "t": random_t,
            "scene_feat": scene_feat,
            "target_feat": target_feat
        }
        estimated_score = self.view_finder(input_data)

        ''' get target score '''
        target_score = - z * std / (std ** 2)

        result = {
            "estimated_score": estimated_score,
            "target_score": target_score,
            "std": std
        }
        return result

    def forward_view(self, data):
        target_pts = data["target_pts"]
        scene_pts = data["scene_pts"]
        
        if self.has_rgb_encoder : 
            if self.rgb_feat_cache:
                rgb_feat = data["rgb_feat"]
            else:
                rgb = data["rgb"]
                rgb_feat = self.rgb_encoder.encode_rgb(rgb)
            if "rgb_feat" not in inspect.signature(self.pts_encoder.encode_points).parameters:
                target_feat = self.pts_encoder.encode_points(target_pts)
                scene_feat = self.pts_encoder.encode_points(scene_pts)
                target_feat = self.fusion_layer(rgb_feat, target_feat)
                scene_feat = self.fusion_layer(rgb_feat, scene_feat)
            else:
                target_feat = self.pts_encoder.encode_points(target_pts, rgb_feat)
                scene_feat = self.pts_encoder.encode_points(scene_pts, rgb_feat)
        else:
            target_feat = self.pts_encoder.encode_points(target_pts)
            scene_feat = self.pts_encoder.encode_points(scene_pts)
        estimated_delta_rot_6d, in_process_sample = self.view_finder.next_best_view(scene_feat, target_feat)
        result = {
            "estimated_delta_rot_6d": estimated_delta_rot_6d,
            "in_process_sample": in_process_sample
        }
        return result


if __name__ == '__main__':
    ConfigManager.load_config_with('../configs/server_train_config.yaml')
    ConfigManager.print_config()
    test_pipeline_config = ConfigManager.get("settings", "pipeline")
    pipeline = Pipeline(test_pipeline_config)
    test_scene = torch.rand(32, 1024, 3).to("cuda:0")
    test_target = torch.rand(32, 1024, 3).to("cuda:0")
    test_delta_rot_6d = torch.rand(32, 6).to("cuda:0")
    a = test_delta_rot_6d[:, :3]
    b = test_delta_rot_6d[:, 3:]
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    normalized_test_delta_rot_6d = torch.cat((a_norm, b_norm), dim=1)
    test_data = {
        'target_pts': test_target,
        'scene_pts': test_scene,
        'delta_rot_6d': normalized_test_delta_rot_6d
    }
    # out_data = pipeline(test_data, "train")
    # print(out_data.keys())
    out_data_test = pipeline(test_data, "test")
    print(out_data_test.keys())
    print(out_data_test["estimated_delta_rot_6d"])
