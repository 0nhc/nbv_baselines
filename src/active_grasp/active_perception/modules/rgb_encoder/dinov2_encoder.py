
import torch
from modules.rgb_encoder.abstract_rgb_encoder import RGBEncoder
from annotations.external_module import external_freeze

@external_freeze
class Dinov2Encoder(RGBEncoder):
    def __init__(self, model_name):
        super(Dinov2Encoder, self).__init__()   
        self.model_name = model_name 
        self.load()
        
    def load(self):
        self.dinov2 = torch.hub.load('modules/module_lib/dinov2', self.model_name, source='local').cuda()

    def encode_rgb(self, rgb):
        with torch.no_grad():
            features_dict = self.dinov2.forward_features(rgb)
            features = features_dict['x_norm_patchtokens']
        return features
