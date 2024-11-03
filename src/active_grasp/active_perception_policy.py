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
import requests
import matplotlib.pyplot as plt
from vgn.grasp import ParallelJawGrasp
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
import tf
from robot_helpers.ros import tf as rhtf
from sklearn.neighbors import NearestNeighbors

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import ros_numpy



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


class ActivePerceptionSingleViewPolicy(SingleViewPolicy):
    def __init__(self, flask_base_url="http://127.0.0.1:5000"):
        super().__init__()
        self.max_views = rospy.get_param("ap_grasp/max_views")
        self.ap_config_path = rospy.get_param("ap_grasp/ap_config_path")
        self.crop_min_radius = rospy.get_param("ap_grasp/crop_min_radius")
        self.crop_radius_step = rospy.get_param("ap_grasp/crop_radius_step")
        self.crop_max_radius = rospy.get_param("ap_grasp/crop_max_radius")
        self.num_knn_neighbours = rospy.get_param("ap_grasp/num_knn_neighbours")
        self.ap_inference_engine = APInferenceEngine(self.ap_config_path)
        self.pcdvis = RealTime3DVisualizer()
        self.updated = False
        self._base_url = flask_base_url

        # For debugging
        self.pcd_publisher = rospy.Publisher('/debug_pcd', PointCloud2, queue_size=10)
        self.grasp_publisher = rospy.Publisher("/grasp_markers", MarkerArray, queue_size=10)


    def request_grasping_pose(self, data):
        response = requests.post(f"{self._base_url}/get_gsnet_grasp", json=data)
        return response.json()
    

    def update(self, img, seg, target_id, support_id, x, q):
        # Visualize scene cloud
        self.vis_scene_cloud(img, x)

        # Visualize Initial Camera Pose
        self.vis_cam_pose(x)

        # When policy hasn't produced an available grasp
        while(self.updated == False):
            # Clear visualization points
            self.publish_pointcloud([[0,0,0]])

            # Inference with our model
            self.target_points, self.scene_points, _ = self.depth_image_to_ap_input(img, 
                                                                                    seg, 
                                                                                    target_id, 
                                                                                    support_id,
                                                                                    scene_sample_num=16384, 
                                                                                    target_sample_num=1024)
            ap_input = {'target_pts': self.target_points,
                        'scene_pts': self.scene_points}
            # save point cloud
            # target_points = self.target_points.cpu().numpy()[0,:,:]
            self.scene_points_cache = self.scene_points.cpu().numpy()[0]
            self.cam_pose_cache = rhtf.lookup(self.base_frame, self.cam_frame, rospy.Time.now()).as_matrix()
            
            self.publish_pointcloud(self.scene_points_cache)

            # time.sleep(10000000)
            # np.savetxt("target_points.txt", target_points, delimiter=",")
            # np.savetxt("scene_points.txt", scene_points, delimiter=",")

            ap_output = self.ap_inference_engine.inference(ap_input)
            delta_rot_6d = ap_output['estimated_delta_rot_6d']

            current_cam_pose = torch.from_numpy(x.as_matrix()).float().to("cuda:0")
            est_delta_rot_mat = self.rotation_6d_to_matrix_tensor_batch(delta_rot_6d)[0]

            target_points_np = self.target_points.cpu().numpy()[0,:,:]
            central_point_of_target = np.mean(target_points_np, axis=0)
            look_at_center = torch.from_numpy(central_point_of_target).float().to("cuda:0")
            # Convert look_at_center's reference frame to arm frame
            look_at_center_T = np.eye(4)
            look_at_center_T[:3, 3] = look_at_center.cpu().numpy()
            look_at_center_T = current_cam_pose.cpu().numpy() @ look_at_center_T
            look_at_center = torch.from_numpy(look_at_center_T[:3, 3]).float().to("cuda:0")
            # Get the NBV
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
            
        # Policy has produced an available nbv and moved to that camera pose
        if(self.updated == True):
            # Request grasping poses from GSNet
            self.target_points, self.scene_points, self.all_points = \
                self.depth_image_to_ap_input(img, seg, target_id, support_id)
            target_points_list = np.asarray(self.target_points.cpu().numpy())[0].tolist()
            central_point_of_target = np.mean(target_points_list, axis=0)

            # Crop points to desired number of points
            num_points_valid = False
            target_points_radius = self.crop_min_radius
            required_num_points = 15000
            # Merge all points and scene points
            self.all_points = np.asarray(self.all_points.cpu().numpy())[0]
            all_points_list = self.all_points.tolist()
            scene_points_list = self.scene_points_cache.tolist()

            # convert scene_points to current frame
            self.cam_pose = rhtf.lookup(self.base_frame, self.cam_frame, rospy.Time.now()).as_matrix()
            for idx, point in enumerate(scene_points_list):
                point = np.asarray(point)
                T_point_cam_old = np.array([[1, 0, 0, point[0]],
                                            [0, 1, 0, point[1]],
                                            [0, 0, 1, point[2]],
                                            [0, 0, 0, 1]])
                # point in arm frame
                T_point_arm = np.matmul(self.cam_pose_cache, T_point_cam_old)
                # point in camera frame
                T_point_cam_new = np.matmul(np.linalg.inv(self.cam_pose),
                                            T_point_arm)
                point = T_point_cam_new[:3, 3].T
                point = point.tolist()
                scene_points_list[idx] = point

            merged_points_list = scene_points_list + all_points_list
            merged_points_list = np.asarray(merged_points_list)
            print("merged_points_list shape: "+str(merged_points_list.shape))

            while not num_points_valid:
                gsnet_input_points = self.crop_pts_sphere(merged_points_list, 
                                                          central_point_of_target, 
                                                          radius=target_points_radius)
                if(len(gsnet_input_points) >= required_num_points):
                    num_points_valid = True
                    # downsample points
                    gsnet_input_points = np.asarray(gsnet_input_points)
                    gsnet_input_points = gsnet_input_points[np.random.choice(gsnet_input_points.shape[0], required_num_points, replace=False)]
                else:
                    target_points_radius += self.crop_radius_step
                    if(target_points_radius > self.crop_max_radius):
                        print("Target points radius exceeds maximum radius")
                        print("Number of points: "+str(len(gsnet_input_points)))
                        print("Interpolating points")
                        # Interpolate points
                        if(len(gsnet_input_points) < self.num_knn_neighbours+1):
                            self.best_grasp = None
                            self.vis.clear_grasp()
                            self.done = True
                            return
                        else:
                            gsnet_input_points = np.asarray(gsnet_input_points)
                            num_interpolation = required_num_points - len(gsnet_input_points)
                            gsnet_input_points = self.interpolate_point_cloud(gsnet_input_points, num_interpolation)
                            num_points_valid = True
            gsnet_input_points = gsnet_input_points.tolist()


            # gsnet_input_points = target_points_list
            # gsnet_input_points = merged_points_list
            self.publish_pointcloud(gsnet_input_points)
            
            # save point cloud as .txt
            np.savetxt("gsnet_input_points.txt", gsnet_input_points, delimiter=",")

            received_points = False
            while(received_points == False): 
                gsnet_grasping_poses = np.asarray(self.request_grasping_pose(gsnet_input_points))
                received_points = True
                print(gsnet_grasping_poses[0].keys())

            # DEBUG: publish grasps
            # self.publish_grasps(gsnet_grasping_poses)

            # Convert all grasping poses' reference frame to arm frame
            current_cam_pose = torch.from_numpy(x.as_matrix()).float().to("cuda:0")
            for gg in gsnet_grasping_poses:
                gg['T'] = current_cam_pose.cpu().numpy().dot(np.asarray(gg['T']))
                # Now here is a mysterous bug, the grasping poses have to be rotated 
                # 90 degrees around y-axis to be in the correct reference frame
                R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                gg['T'][:3, :3] = gg['T'][:3, :3].dot(R)
            
            # Convert grasping poses to ParallelJawGrasp objects
            grasps = []
            qualities = []
            for gg in gsnet_grasping_poses:
                T = Transform.from_matrix(np.asarray(gg['T']))
                width = 0.075
                grasp = ParallelJawGrasp(T, width)
                grasps.append(grasp)
                qualities.append(gg['score'])
            
            # Visualize grasps
            # self.vis.grasps(self.base_frame, grasps, qualities)
            
            # Filter grasps
            filtered_grasps = []
            filtered_qualities = []
            for grasp, quality in zip(grasps, qualities):
                pose = grasp.pose
                # tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
                tip = pose.translation
                if self.bbox.is_inside(tip):
                # if(True):
                    q_grasp = self.solve_ee_ik(q, pose * self.T_grasp_ee)
                    if q_grasp is not None:
                        filtered_grasps.append(grasp)
                        filtered_qualities.append(quality)
            if len(filtered_grasps) > 0:
                self.best_grasp, quality = self.select_best_grasp(filtered_grasps, filtered_qualities)
                # self.vis.clear_grasps()
                self.vis.grasp(self.base_frame, self.best_grasp, quality)
            else:
                self.best_grasp = None
                self.vis.clear_grasp()
            self.done = True
            # self.publish_pointcloud([[0,0,0]])
    
    def publish_grasps(self, gg):
        marker_array = MarkerArray()
        marker_array.markers = []
        for idx, g in enumerate(gg):
            g['T'] = np.asarray(g['T'])
            marker = Marker()
            marker.header.frame_id = "camera_depth_optical_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "grasps"
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = g['T'][0, 3]
            marker.pose.position.y = g['T'][1, 3]
            marker.pose.position.z = g['T'][2, 3]
            q = tf.transformations.quaternion_from_matrix(g['T'])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = 0.1
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.grasp_publisher.publish(marker_array)

    def publish_pointcloud(self, point_cloud):
        point_cloud = np.asarray(point_cloud)
        cloud_msg = self.create_pointcloud_msg(point_cloud)
        self.pcd_publisher.publish(cloud_msg)

    def create_pointcloud_msg(self, point_cloud):
        # Define the header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_depth_optical_frame'  # Change this to your desired frame of reference

        # Define the fields for the PointCloud2 message
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create the PointCloud2 message
        cloud_msg = pc2.create_cloud(header, fields, point_cloud)

        return cloud_msg

    def crop_pts_sphere(self, points, crop_center, radius=0.2):
        crop_mask = np.linalg.norm(points - crop_center, axis=1) < radius
        return points[crop_mask].tolist()
    
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
    
        # pose = rhtf.lookup(self.base_frame, "panda_link8", rospy.Time.now())  
        # ee_matrix = pose.as_matrix()
        # minimum_difference = np.inf
        # for i in range(len(grasps)-1, -1, -1):
        #     grasp = grasps[i]
        #     g_matrix = grasp.pose.as_matrix()
        #     # calculatr the Frobenius norm (rotation difference)
        #     rotation_difference = np.linalg.norm(ee_matrix[:3, :3] - g_matrix[:3, :3])
        #     if rotation_difference < minimum_difference:
        #         minimum_difference = rotation_difference
        #         best_grasp = grasp
        #         best_quality = qualities[i]
        # return best_grasp, best_quality

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

    def depth_image_to_ap_input(self, depth_img, seg_img, target_id, support_id,
                                scene_sample_num=-1, target_sample_num=-1):
        target_points = []
        scene_points = []
        all_points = []

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
                
                if(int(seg_id) != int(255)): # no background points
                    all_points.append([x,y,z])
                    if(int(seg_id) != int(support_id)): # no support points
                        # This pixel belongs to the scene
                        scene_points.append([x,y,z])
                        if(int(seg_id) == int(target_id)):
                            # This pixel belongs to the target object to be grasped
                            target_points.append([x,y,z])
        
        # Sample points
        all_points = np.asarray(all_points)
        target_points = np.asarray(target_points)
        scene_points = np.asarray(scene_points)
        if scene_sample_num > 0:
            if scene_points.shape[0] < scene_sample_num:
                print("Scene points are less than the required sample number")
                num_interpolation = scene_sample_num - scene_points.shape[0]
                scene_points = self.interpolate_point_cloud(scene_points, num_interpolation)
                print("Interpolated scene points. Shape: "+str(scene_points.shape))
            else:
                scene_points = scene_points[np.random.choice(scene_points.shape[0], scene_sample_num, replace=False)]

        if target_sample_num > 0:
            if target_points.shape[0] < target_sample_num:
                print("Target points are less than the required sample number")
                num_interpolation = target_sample_num - target_points.shape[0]
                target_points = self.interpolate_point_cloud(target_points, num_interpolation)
                print("Interpolated target points. Shape: "+str(target_points.shape))
            else:
                target_points = target_points[np.random.choice(target_points.shape[0], target_sample_num, replace=False)]

        # reshape points
        target_points = target_points.reshape(1, target_points.shape[0], 3)
        # self.pcdvis.update_points(target_points)
        target_points = torch.from_numpy(target_points).float().to("cuda:0")
        
        scene_points = scene_points.reshape(1, scene_points.shape[0], 3)
        scene_points = torch.from_numpy(scene_points).float().to("cuda:0")

        all_points = all_points.reshape(1, all_points.shape[0], 3)
        all_points = torch.from_numpy(all_points).float().to("cuda:0")
        
        return target_points, scene_points, all_points

    def interpolate_point_cloud(self, points, num_new_points):
        # Fit NearestNeighbors on existing points
        nbrs = NearestNeighbors(n_neighbors=self.num_knn_neighbours).fit(points)
        interpolated_points = []
        
        for _ in range(num_new_points):
            random_point = points[np.random.choice(len(points))]
            distances, indices = nbrs.kneighbors([random_point])
            neighbors = points[indices[0]]
            new_point = neighbors.mean(axis=0)  # Average of neighbors
            interpolated_points.append(new_point)
        
        return np.vstack([points, interpolated_points])