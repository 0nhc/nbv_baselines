from abc import abstractmethod

from torch import nn


class PointsEncoder(nn.Module):
    def __init__(self):
        super(PointsEncoder, self).__init__()

    @abstractmethod
    def encode_points(self, pts):
        pass
