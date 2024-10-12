import sys
import os
path = os.path.abspath(__file__)
for i in range(3):
    path = os.path.dirname(path)
PROJECT_ROOT = path
sys.path.append(PROJECT_ROOT)

from modules.rgb_encoder.abstract_rgb_encoder import RGBEncoder
from modules.rgb_encoder.dinov2_encoder import Dinov2Encoder


class RGBEncoderFactory:
    @staticmethod
    def create(name, config) -> RGBEncoder:
        general_config = config["general"]
        rgb_encoder_config = config["rgb_encoder"][name]
        if name == "dinov2":
            return Dinov2Encoder(
                model_name=rgb_encoder_config["model_name"]
            )
        else:
            raise ValueError(f"Unknown encoder name: {name}")


''' ------------ Debug ------------ '''
if __name__ == "__main__":
    from configs.config import ConfigManager
    import torch
    from PIL import Image
    import cv2
    from torchvision import transforms
    ConfigManager.load_config_with('configs/local_train_config.yaml')
    ConfigManager.print_config()
    image_size = 480
    path = "/mnt/h/BaiduSyncdisk/workspace/ws_active_pose/project/ActivePerception/test/img0.jpg"
    img = cv2.imread(path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([           
                        transforms.Resize(image_size),
                        transforms.CenterCrop(int(image_size//14)*14),              
                        transforms.ToTensor(),                    
                        transforms.Normalize(mean=0.5, std=0.2)
                        ])
    
    rgb = transform(img)
    print(rgb.shape)
    rgb_encoder = RGBEncoderFactory.create(name="dinov2", config=ConfigManager.get("modules"))
    rgb_encoder.load()
    print(rgb_encoder)
    rgb = rgb.to("cuda:0")
    rgb = rgb.unsqueeze(0)
    rgb_encoder = rgb_encoder.to("cuda:0")
    
    rgb_feat = rgb_encoder.encode_rgb(rgb)

    print(rgb_feat.shape)
    rgb_encoder.visualize_features(rgb_feat[0])
