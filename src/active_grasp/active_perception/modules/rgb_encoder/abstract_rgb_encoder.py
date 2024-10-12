from abc import abstractmethod
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class RGBEncoder(nn.Module):
    def __init__(self):
        super(RGBEncoder, self).__init__()

    @abstractmethod
    def encode_rgb(self, rgb):
        pass

    @staticmethod
    def visualize_features(features, save_path=None):
        patch,feat_dim = features.shape
        patch_h = int(patch ** 0.5)
        patch_w = patch_h
        total_features = features.reshape(patch_h * patch_w, feat_dim)
        pca = PCA(n_components=3)
        if isinstance(total_features, torch.Tensor):
            total_features = total_features.cpu().numpy()
        pca.fit(total_features)
        pca_features = pca.transform(total_features)
        pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                     (pca_features[:, 0].max() - pca_features[:, 0].min())
        plt.subplot(1, 3, 1)
        plt.imshow(pca_features[:,0].reshape(patch_h, patch_w))
        pca_features_bg = pca_features[:, 0] > 0.5 # from first histogram
        pca_features_fg = np.ones_like(pca_features_bg)
        plt.subplot(1, 3, 2)
        plt.imshow(pca_features_bg.reshape(patch_h, patch_w))
        pca.fit(total_features[pca_features_fg]) 
        pca_features_left = pca.transform(total_features[pca_features_fg])
        for i in range(3):
            pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

        pca_features_rgb = pca_features.copy()
        pca_features_rgb[pca_features_bg] = 0
        pca_features_rgb[pca_features_fg] = pca_features_left
        pca_features_rgb = pca_features_rgb.reshape(1, patch_h, patch_w, 3)
        
        plt.subplot(1, 3, 3)
        if save_path:
            plt.imsave(save_path, pca_features_rgb[0])
        else:
            plt.imshow(pca_features_rgb[0])
            plt.show()