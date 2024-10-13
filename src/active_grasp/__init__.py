from .policy import register
from .baselines import *
from .nbv import NextBestView
from.active_perception_policy import ActivePerceptionPolicy

register("initial-view", InitialView)
register("top-view", TopView)
register("top-trajectory", TopTrajectory)
register("fixed-trajectory", FixedTrajectory)
register("nbv", NextBestView)
register("ap", ActivePerceptionPolicy)