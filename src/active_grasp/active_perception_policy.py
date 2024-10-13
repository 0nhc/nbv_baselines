import itertools
from numba import jit
import numpy as np
import rospy
from .policy import MultiViewPolicy, SingleViewPolicy
from .timer import Timer
from .active_perception_demo import APInferenceEngine
from robot_helpers.spatial import Transform
import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt


class RealTime3DVisualizer:
    def __init__(self):
        points = np.random.rand(1, 1, 3)
        self.points = points[0]  # Extract the points (n, 3)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initial plot setup
        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='b', marker='o')

        # Set labels for each axis
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set title
        self.ax.set_title('Real-time 3D Points Visualization')

        # Show the plot in interactive mode
        plt.ion()
        plt.show()


    def update_points(self, new_points):
        # Ensure the points have the expected shape (1, n, 3)
        assert new_points.shape[0] == 1 and new_points.shape[2] == 3, "Input points must have shape (1, n, 3)"

        # Update the stored points
        self.points = new_points[0]  # Extract the points (n, 3)

        # Remove the old scatter plot and draw new points
        self.scatter.remove()
        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='b', marker='o')

        # Pause briefly to allow the plot to update
        plt.pause(0.001)


class ActivePerceptionMultiViewPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.max_views = rospy.get_param("ap_grasp/max_views")
        self.ap_config_path = rospy.get_param("ap_grasp/ap_config_path")
        self.max_inference_num = rospy.get_param("ap_grasp/max_inference_num")
        self.ap_inference_engine = APInferenceEngine(self.ap_config_path)
        self.pcdvis = RealTime3DVisualizer()


    def update(self, img, seg, target_id, x, q):
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("state_update"):
                self.integrate(img, x, q)

            # When policy hasn't produced an available grasp
            c = 0
            while(c < self.max_inference_num):
                # Inference with our model
                target_points, scene_points = self.depth_image_to_ap_input(img, seg, target_id)
                ap_input = {'target_pts': target_points,
                            'scene_pts': scene_points}
                ap_output = self.ap_inference_engine.inference(ap_input)
                c += 1
                delta_rot_6d = ap_output['estimated_delta_rot_6d']

                current_cam_pose = torch.from_numpy(x.as_matrix()).float().to("cuda:0")
                est_delta_rot_mat = self.rotation_6d_to_matrix_tensor_batch(delta_rot_6d)[0]
                look_at_center = torch.from_numpy(self.bbox.center).float().to("cuda:0")
                nbv_tensor = self.get_transformed_mat(current_cam_pose, 
                                                        est_delta_rot_mat, 
                                                        look_at_center)
                nbv_numpy = nbv_tensor.cpu().numpy()
                nbv_transform = Transform.from_matrix(nbv_numpy)
                x_d = nbv_transform

                # Check if this pose available
                if(self.solve_cam_ik(self.q0, x_d)):
                    self.x_d = x_d
                    self.updated = True
                    print("Found an NBV!")
                    break
    

    def vis_cam_pose(self, x):
        # Integrate
        self.views.append(x)
        self.vis.path(self.base_frame, self.intrinsic, self.views)
    
    def vis_scene_cloud(self, img, x):
        self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
        scene_cloud = self.tsdf.get_scene_cloud()
        self.vis.scene_cloud(self.task_frame, np.asarray(scene_cloud.points))

    def rotation_6d_to_matrix_tensor_batch(self, d6: torch.Tensor) -> torch.Tensor:
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    

    def get_transformed_mat(self, src_mat, delta_rot, target_center_w):
        src_rot = src_mat[:3, :3] 
        dst_rot = src_rot @ delta_rot.T
        dst_mat = torch.eye(4).to(dst_rot.device)
        dst_mat[:3, :3] = dst_rot
        distance = torch.norm(target_center_w - src_mat[:3, 3])
        z_axis_camera = dst_rot[:3, 2].reshape(-1)
        new_camera_position_w = target_center_w - distance * z_axis_camera
        dst_mat[:3, 3] = new_camera_position_w
        return dst_mat

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
        self.pcdvis.update_points(target_points)
        target_points = torch.from_numpy(target_points).float().to("cuda:0")
        scene_points = np.asarray(scene_points)
        scene_points = scene_points.reshape(1, scene_points.shape[0], 3)
        scene_points = torch.from_numpy(scene_points).float().to("cuda:0")
        
        return target_points, scene_points


    def best_grasp_prediction_is_stable(self):
        if self.best_grasp:
            t = (self.T_task_base * self.best_grasp.pose).translation
            i, j, k = (t / self.tsdf.voxel_size).astype(int)
            qs = self.qual_hist[:, i, j, k]
            if np.count_nonzero(qs) == self.T and np.mean(qs) > 0.9:
                return True
        return False


class ActivePerceptionSingleViewPolicy(SingleViewPolicy):
    def __init__(self):
        super().__init__()
        self.max_views = rospy.get_param("ap_grasp/max_views")
        self.ap_config_path = rospy.get_param("ap_grasp/ap_config_path")
        self.ap_inference_engine = APInferenceEngine(self.ap_config_path)
        self.pcdvis = RealTime3DVisualizer()
        self.updated = False


    def update(self, img, seg, target_id, x, q):
        # Visualize scene cloud
        self.vis_scene_cloud(img, x)

        # Visualize Initial Camera Pose
        self.vis_cam_pose(x)

        # When policy hasn't produced an available grasp
        while(self.updated == False):
            # Inference with our model
            target_points, scene_points = self.depth_image_to_ap_input(img, seg, target_id)
            ap_input = {'target_pts': target_points,
                        'scene_pts': scene_points}
            ap_output = self.ap_inference_engine.inference(ap_input)
            delta_rot_6d = ap_output['estimated_delta_rot_6d']

            current_cam_pose = torch.from_numpy(x.as_matrix()).float().to("cuda:0")
            est_delta_rot_mat = self.rotation_6d_to_matrix_tensor_batch(delta_rot_6d)[0]
            look_at_center = torch.from_numpy(self.bbox.center).float().to("cuda:0")
            nbv_tensor = self.get_transformed_mat(current_cam_pose, 
                                                    est_delta_rot_mat, 
                                                    look_at_center)
            nbv_numpy = nbv_tensor.cpu().numpy()
            nbv_transform = Transform.from_matrix(nbv_numpy)
            x_d = nbv_transform

            # Check if this pose available
            if(self.solve_cam_ik(self.q0, x_d)):
                self.vis_cam_pose(x_d)
                self.x_d = x_d
                self.updated = True
                print("Found an NBV!")
                return
        # Policy has produced an available grasp
        if(self.updated == True):
            self.generate_grasp(q)
            self.done = True
            
    def deactivate(self):
        self.vis.clear_ig_views()
        self.updated = False

    def vis_cam_pose(self, x):
        # Integrate
        self.views.append(x)
        self.vis.path(self.base_frame, self.intrinsic, self.views)
    
    def vis_scene_cloud(self, img, x):
        self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
        scene_cloud = self.tsdf.get_scene_cloud()
        self.vis.scene_cloud(self.task_frame, np.asarray(scene_cloud.points))

    def generate_grasp(self, q):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        self.vis.quality(self.task_frame, self.tsdf.voxel_size, out.qual, 0.9)
        grasps, qualities = self.filter_grasps(out, q)

        if len(grasps) > 0:
            self.best_grasp, quality = self.select_best_grasp(grasps, qualities)
            self.vis.grasp(self.base_frame, self.best_grasp, quality)
        else:
            self.best_grasp = None
            self.vis.clear_grasp()


    def select_best_grasp(self, grasps, qualities):
        i = np.argmax(qualities)
        return grasps[i], qualities[i]

    def rotation_6d_to_matrix_tensor_batch(self, d6: torch.Tensor) -> torch.Tensor:
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    

    def get_transformed_mat(self, src_mat, delta_rot, target_center_w):
        src_rot = src_mat[:3, :3] 
        dst_rot = src_rot @ delta_rot.T
        dst_mat = torch.eye(4).to(dst_rot.device)
        dst_mat[:3, :3] = dst_rot
        distance = torch.norm(target_center_w - src_mat[:3, 3])
        z_axis_camera = dst_rot[:3, 2].reshape(-1)
        new_camera_position_w = target_center_w - distance * z_axis_camera
        dst_mat[:3, 3] = new_camera_position_w
        return dst_mat

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
        self.pcdvis.update_points(target_points)
        target_points = torch.from_numpy(target_points).float().to("cuda:0")
        scene_points = np.asarray(scene_points)
        scene_points = scene_points.reshape(1, scene_points.shape[0], 3)
        scene_points = torch.from_numpy(scene_points).float().to("cuda:0")
        
        return target_points, scene_points
