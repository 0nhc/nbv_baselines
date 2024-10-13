import itertools
from numba import jit
import numpy as np
import rospy
from .policy import MultiViewPolicy
from .timer import Timer

from .active_perception_demo import APInferenceEngine


class ActivePerceptionPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.max_views = rospy.get_param("ap_grasp/max_views")
        self.ap_config_path = rospy.get_param("ap_grasp/ap_config_path")
        self.ap_inference_engine = APInferenceEngine(self.ap_config_path)

    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)

    def update(self, img, seg, target_id, x, q):
        target_points, scene_points = self.depth_image_to_ap_input(img, seg, target_id)
        # if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
        #     self.done = True
        # else:
        #     with Timer("state_update"):
        #         self.integrate(img, x, q)
        #     with Timer("view_generation"):
        #         views = self.generate_views(q)
        #     with Timer("ig_computation"):
        #         gains = [self.ig_fn(v, self.downsample) for v in views]
        #     with Timer("cost_computation"):
        #         costs = [self.cost_fn(v) for v in views]
        #     utilities = gains / np.sum(gains) - costs / np.sum(costs)
        #     self.vis.ig_views(self.base_frame, self.intrinsic, views, utilities)
        #     i = np.argmax(utilities)
        #     nbv, gain = views[i], gains[i]

        #     if gain < self.min_gain and len(self.views) > self.T:
        #         self.done = True

        #     self.x_d = nbv
    
    def depth_image_to_ap_input(self, depth_img, seg_img, target_id):
        target_points = []
        scene_points = []

        K = self.intrinsic.K
        depth_shape = depth_img.shape
        seg_shape = seg_img.shape
        if(depth_shape == seg_shape):
            img_shape = depth_shape
        else:
            print("Depth image shape and segmentation image shape are not the same")
            return None
        
        # Convert depth image to 3D points
        u_indices , v_indices = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
        x_factors = (u_indices - K[0, 2]) / K[0, 0]
        y_factors = (v_indices - K[1, 2]) / K[1, 1]
        z_mat = depth_img
        x_mat = x_factors * z_mat
        y_mat = y_factors * z_mat
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                seg_id = seg_img[i, j]
                x = x_mat[i][j]
                y = y_mat[i][j]
                z = z_mat[i][j]
                if(int(seg_id) == int(target_id)):
                    # This pixel belongs to the target object to be grasped
                    target_points.append([x,y,z])
                else:
                    # This pixel belongs to the scene
                    scene_points.append([x,y,z])
        
        target_points = np.asarray(target_points)
        target_points = target_points.reshape(1, target_points.shape[0], 3)
        scene_points = np.asarray(scene_points)
        scene_points = scene_points.reshape(1, scene_points.shape[0], 3)
        
        return target_points, scene_points


    def best_grasp_prediction_is_stable(self):
        if self.best_grasp:
            t = (self.T_task_base * self.best_grasp.pose).translation
            i, j, k = (t / self.tsdf.voxel_size).astype(int)
            qs = self.qual_hist[:, i, j, k]
            if np.count_nonzero(qs) == self.T and np.mean(qs) > 0.9:
                return True
        return False

