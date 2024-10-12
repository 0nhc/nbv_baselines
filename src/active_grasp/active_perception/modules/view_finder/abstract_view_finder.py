from abc import abstractmethod

from torch import nn


class ViewFinder(nn.Module):
    def __init__(self):
        super(ViewFinder, self).__init__()

    @abstractmethod
    def next_best_view(self, scene_pts_feat, target_pts_feat):
        pass
