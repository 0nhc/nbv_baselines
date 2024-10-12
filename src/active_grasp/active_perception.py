import os
import sys
import numpy as np
import torch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



path = os.path.abspath(__file__)
for i in range(2):
    path = os.path.dirname(path)
PROJECT_ROOT = path
sys.path.append(PROJECT_ROOT)

from active_perception.configs.config import ConfigManager
from active_perception.modules.pipeline import Pipeline

class InferenceEngine():
    RESULTS_DIR_NAME: str = 'results'
    LOG_DIR_NAME: str = 'log'

    def __init__(self, config_path):
        ''' Config Manager '''
        ConfigManager.load_config_with(config_path)

        ''' Pytorch Seed '''
        seed = ConfigManager.get("settings", "general", "seed")
        np.random.seed(seed)
        torch.manual_seed(seed)

        ''' Pipeline '''
        # self.pipeline_config = {'pts_encoder': 'pointnet', 'view_finder': 'gradient_field'}
        self.pipeline_config = ConfigManager.get("settings", "pipeline")
        self.device = ConfigManager.get("settings", "general", "device")
        self.pipeline = Pipeline(self.pipeline_config)
        self.parallel = ConfigManager.get("settings","general","parallel")
        if self.parallel and self.device == "cuda":
            self.pipeline = torch.nn.DataParallel(self.pipeline)
        self.pipeline = self.pipeline.to(self.device)

        ''' Experiment '''
        # self.model_path = '~/Downloads/full_149_241009.pth'
        self.model_path = ConfigManager.get("settings", "experiment", "model_path")
        self.load(self.model_path)

    
    def load(self, path):
        state_dict = torch.load(path)
        if self.parallel:
            self.pipeline.module.load_state_dict(state_dict)
        else:
            self.pipeline.load_state_dict(state_dict)


    def inference(self, data):
        self.pipeline.eval()
        with torch.no_grad():
            output = self.pipeline(data, Pipeline.TEST_MODE)
        
        return output


if __name__ == "__main__":
    ''' Load Configs '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=PROJECT_ROOT+"/active_grasp/active_perception/configs/local_inference_config.yaml")
    args = parser.parse_args()

    ''' Initialize Test Data '''
    test_scene = torch.rand(1, 1024, 3).to("cuda:0")
    test_target = torch.rand(1, 1024, 3).to("cuda:0")
    test_delta_rot_6d = torch.rand(1, 6).to("cuda:0")
    a = test_delta_rot_6d[:, :3]
    b = test_delta_rot_6d[:, 3:]
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    normalized_test_delta_rot_6d = torch.cat((a_norm, b_norm), dim=1)
    test_data = {
        'target_pts': test_target,
        'scene_pts': test_scene,
    }

    ''' Inference '''
    infenrence_engine = InferenceEngine(args.config)
    output = infenrence_engine.inference(test_data)
    print(output.keys())
    print(output['estimated_delta_rot_6d'])