import numpy as np
import pickle
import json
import pickle
import cv2
import os
import re
from scipy.spatial.transform import Rotation as R

class DepthToPCL:

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "Use init_from_disk or init_from_memory to create an instance"
        )

    @classmethod
    def _initialize(
        cls,
        distance_to_camera_path=None,
        rgb_path=None,
        camera_params_path=None,
        seg_path=None,
        seg_label_path=None,
        depth=None,
        rgb=None,
        seg=None,
        seg_label=None,
        camera_params=None,
    ):
        instance = super().__new__(cls)
        instance._distance_to_camera_path = distance_to_camera_path
        instance._rgb_path = rgb_path
        instance._camera_params_path = camera_params_path
        instance._seg_path = seg_path
        instance._seg_label_path = seg_label_path
        instance._depth = depth
        instance._rgb = rgb
        instance._seg = seg
        instance._seg_label = seg_label
        instance._camera_params = camera_params

        if any(
            path is not None
            for path in [
                distance_to_camera_path,
                rgb_path,
                camera_params_path,
                seg_path,
                seg_label_path,
            ]
        ):
            instance._load_from_disk()

        instance._setup()
        return instance

    @classmethod
    def init_from_disk(
        cls,
        distance_to_camera_path,
        rgb_path,
        camera_params_path,
        seg_path,
        seg_label_path,
    ):
        return cls._initialize(
            distance_to_camera_path=distance_to_camera_path,
            rgb_path=rgb_path,
            camera_params_path=camera_params_path,
            seg_path=seg_path,
            seg_label_path=seg_label_path,
        )

    @classmethod
    def init_from_memory(cls, depth, rgb, seg, seg_label, camera_params):
        return cls._initialize(
            depth=depth,
            rgb=rgb,
            seg=seg,
            seg_label=seg_label,
            camera_params=camera_params,
        )

    def _load_from_disk(self):
        self._depth = np.load(self._distance_to_camera_path)
        self._seg = cv2.imread(self._seg_path, cv2.IMREAD_UNCHANGED)

        with open(self._seg_label_path, "r") as f:
            self._seg_label = json.load(f)
        with open(self._camera_params_path) as f:
            self._camera_params = json.load(f)

    def _setup(self):
        self._read_camera_params()
        self._get_intrinsic_matrix()

    def _read_camera_params(self):
        self._h_aperture = self._camera_params["cameraAperture"][0]
        self._v_aperture = self._camera_params["cameraAperture"][1]
        self._h_aperture_offset = self._camera_params["cameraApertureOffset"][0]
        self._v_aperture_offset = self._camera_params["cameraApertureOffset"][1]
        self._focal_length = self._camera_params["cameraFocalLength"]
        self._h_resolution = self._camera_params["renderProductResolution"][0]
        self._v_resolution = self._camera_params["renderProductResolution"][1]
        self._cam_t = self._camera_params["cameraViewTransform"]

    def _get_intrinsic_matrix(self):
        self._focal_x = self._h_resolution * self._focal_length / self._h_aperture
        self._focal_y = self._v_resolution * self._focal_length / self._v_aperture
        self._center_x = self._h_resolution / 2
        self._center_y = self._v_resolution / 2
        self.intrinsic_matrix = np.array(
            [
                [self._focal_x, 0, self._center_x],
                [0, self._focal_y, self._center_y],
                [0, 0, 1],
            ]
        )
        return self.intrinsic_matrix

    def _get_extrinsic_matrix(self):
        self._cam_pose = np.linalg.inv(np.resize(self._cam_t, (4, 4))).T.dot(
            np.mat([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1.0], [0, 0, 0, 1]])
        )
        return self._cam_pose

    

    def get_pcd(self, target_name=None):
        u_indices, v_indices = np.meshgrid(
            np.arange(self._h_resolution), np.arange(self._v_resolution)
        )
        x_factors = (u_indices - self._center_x) / self._focal_x
        y_factors = (v_indices - self._center_y) / self._focal_y
        if target_name is not None:
            if target_name == OmniUtil.FOREGROUND:
                unlabelled_mask = self.get_mask_rgba(
                    self._seg_label, OmniUtil.UNLABELLED
                )
                background_mask = self.get_mask_rgba(
                    self._seg_label, OmniUtil.BACKGROUND
                )
                if unlabelled_mask is None:
                    target_mask = (self._seg != background_mask).any(axis=2)
                else:
                    target_mask = (self._seg != unlabelled_mask).any(axis=2) & (
                        self._seg != background_mask
                    ).any(axis=2)
            else:
                target_mask = (
                    self._seg == self.get_mask_rgba(self._seg_label, target_name)
                ).all(axis=2)
        else:
            target_mask = np.ones((self._v_resolution, self._h_resolution), dtype=bool)
        valid_x_factors = x_factors[target_mask]
        valid_y_factors = y_factors[target_mask]
        valid_z_factors = self._depth[target_mask]
        points = np.stack([valid_x_factors, valid_y_factors, valid_z_factors], axis=1)
        return points
    
    @staticmethod
    def get_mask_rgba(mask_labels, mask_name):
        name_list = [name_dict["class"] for name_dict in list(mask_labels.values())]
        if mask_name not in name_list:
            return None
        rgba_list = list(mask_labels.keys())
        mask_rgba_str = rgba_list[name_list.index(mask_name)]
        r, g, b, a = re.findall("\d+", mask_rgba_str)
        r, g, b, a = int(b), int(g), int(r), int(a)
        return r, g, b, a

    def get_segmented_pcd(self, target_list, N=15000):
        u_indices, v_indices = np.meshgrid(
            np.arange(self._h_resolution), np.arange(self._v_resolution)
        )
        x_factors = (u_indices - self._center_x) / self._focal_x
        y_factors = (v_indices - self._center_y) / self._focal_y
        points_dict = {}
        total_points_with_label = []
        for target_idx in range(len(target_list)):
            target_name = target_list[target_idx]
            target_mask = (
                self._seg == self.get_mask_rgba(self._seg_label, target_name)
            ).all(axis=2)
            valid_x_factors = x_factors[target_mask]
            valid_y_factors = y_factors[target_mask]
            valid_z_factors = self._depth[target_mask]
            label = np.ones_like(valid_x_factors) * target_idx
            target_points_with_label = np.stack(
                [valid_x_factors, valid_y_factors, valid_z_factors, label], axis=1
            )
            total_points_with_label.append(target_points_with_label)
        total_points_with_label = np.concatenate(total_points_with_label, axis=0)
        total_points_with_label = self.sample_pcl(total_points_with_label, N)
        total_points = total_points_with_label[:, :3]
        for target_idx in range(len(target_list)):
            target_name = target_list[target_idx]
            pts_seg = total_points_with_label[:, 3] == target_idx
            points_dict[target_name] = total_points_with_label[pts_seg, :3]

        return total_points, points_dict

    def get_rgb(self):
        return self._rgb

    @staticmethod
    def sample_pcl(pcl, n_pts=1024):
        indices = np.random.choice(pcl.shape[0], n_pts, replace=pcl.shape[0] < n_pts)
        return pcl[indices, :]


class OmniUtil:
    FOREGROUND = "FOREGROUND"
    BACKGROUND = "BACKGROUND"
    UNLABELLED = "UNLABELLED"
    NON_OBJECT_LIST = ['chair_028', 'chair_029', 'chair_026', 'chair_027', 'table_025', 'table_027', 'table_026', 'table_028', 'sofa_014', 'sofa_013', 'picnic_basket_010', 'picnic_basket_011', 'cabinet_009', 'flower_pot_023', 'flower_pot_022', 'flower_pot_021', 'chair_017', 'chair_020', 'chair_012', 'chair_010', 'chair_018', 'chair_025', 'chair_024', 'chair_011', 'chair_001', 'chair_013', 'chair_004', 'chair_021', 'chair_023', 'chair_006', 'chair_014', 'chair_007', 'chair_003', 'chair_009', 'chair_022', 'chair_015', 'chair_016', 'chair_008', 'chair_005', 'chair_019', 'chair_002', 'table_004', 'table_023', 'table_014', 'table_024', 'table_019', 'table_022', 'table_007', 'table_017', 'table_013', 'table_002', 'table_016', 'table_009', 'table_008', 'table_003', 'table_015', 'table_001', 'table_018', 'table_005', 'table_020', 'table_021', 'sofa_001', 'sofa_005', 'sofa_012', 'sofa_009', 'sofa_006', 'sofa_008', 'sofa_011', 'sofa_004', 'sofa_003', 'sofa_002', 'sofa_007', 'sofa_010', 'picnic_basket_005', 'picnic_basket_004', 'picnic_basket_001', 'picnic_basket_008', 'picnic_basket_002', 'picnic_basket_009', 'picnic_basket_006', 'picnic_basket_003', 'picnic_basket_007', 'cabinet_006', 'cabinet_008', 'cabinet_002', 'cabinet_001', 'cabinet_005', 'cabinet_007', 'flower_pot_013', 'flower_pot_005', 'flower_pot_008', 'flower_pot_001', 'flower_pot_003', 'flower_pot_020', 'flower_pot_006', 'flower_pot_012', 'flower_pot_018', 'flower_pot_007', 'flower_pot_002', 'flower_pot_011', 'flower_pot_010', 'flower_pot_016', 'flower_pot_004', 'flower_pot_014', 'flower_pot_017', 'flower_pot_019']
    CAMERA_PARAMS_TEMPLATE = "camera_params_{}.json"
    DISTANCE_TEMPLATE = "distance_to_image_plane_{}.npy"
    RGB_TEMPLATE = "rgb_{}.png"
    MASK_TEMPLATE = "semantic_segmentation_{}.png"
    MASK_LABELS_TEMPLATE = "semantic_segmentation_labels_{}.json"
    SCORE_LABEL_TEMPLATE = "label_{}.json"
    RGB_FEAT_TEMPLATE = "rgb_feat_{}.npy"

    @staticmethod
    def get_depth_to_pointcloud_instance(path):
        root, idx = path[:-4], path[-4:]
        distance2plane_path = os.path.join(root, OmniUtil.DISTANCE_TEMPLATE.format(idx))
        rgb_path = os.path.join(root, OmniUtil.RGB_TEMPLATE.format(idx))
        cam_params_path = os.path.join(
            root, OmniUtil.CAMERA_PARAMS_TEMPLATE.format(idx)
        )
        seg_path = os.path.join(root, OmniUtil.MASK_TEMPLATE.format(idx))
        seg_labels_path = os.path.join(root, OmniUtil.MASK_LABELS_TEMPLATE.format(idx))
        depth_to_pcd = DepthToPCL.init_from_disk(
            distance2plane_path, rgb_path, cam_params_path, seg_path, seg_labels_path
        )
        return depth_to_pcd

    @staticmethod
    def get_points(path, object_name=None):
        depth_to_pcd = OmniUtil.get_depth_to_pointcloud_instance(path)
        pcd = depth_to_pcd.get_pcd(object_name)
        points = np.asarray(pcd, dtype=np.float32)
        return points

    @staticmethod
    def get_segmented_points(path, target_list):
        depth_to_pcd = OmniUtil.get_depth_to_pointcloud_instance(path)
        total_points, target_points_dict = depth_to_pcd.get_segmented_pcd(target_list)
        return total_points, target_points_dict

    @staticmethod
    def get_object_list(path, contains_non_obj=False):
        root, idx = path[:-4], path[-4:]
        seg_labels_path = os.path.join(root, OmniUtil.MASK_LABELS_TEMPLATE.format(idx))
        with open(seg_labels_path, "r") as f:
            seg_labels = json.load(f)
        object_list = [v["class"] for v in seg_labels.values()]
        
        object_list.remove(OmniUtil.BACKGROUND)
        if OmniUtil.UNLABELLED in object_list:
            object_list.remove(OmniUtil.UNLABELLED)
        occluder_list = pickle.load(open(os.path.join(root,"occluder.pickle"), "rb"))
        fall_objects_list = pickle.load(open(os.path.join(root,"fall_objects.pickle"), "rb"))
        non_obj_list = occluder_list + fall_objects_list
        if not contains_non_obj:
            for non_obj in non_obj_list:
                if non_obj in object_list:
                    object_list.remove(non_obj)
        return object_list

    @staticmethod
    def get_rotation_mat(path):
        root, idx = os.path.split(path)
        camera_params_path = os.path.join(
            root, OmniUtil.CAMERA_PARAMS_TEMPLATE.format(idx)
        )
        with open(camera_params_path, "r") as f:
            raw_camera_params = json.load(f)
        cam_transform = np.asarray(raw_camera_params["cameraViewTransform"]).reshape(
            (4, 4)
        )
        cam_rot_mat = cam_transform[:3, :3].dot(
            np.mat([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        )
        return cam_rot_mat

    @staticmethod
    def get_rgb(path):
        root, idx = os.path.split(path)
        rgb_path = os.path.join(root, OmniUtil.RGB_TEMPLATE.format(idx))
        rgb = cv2.imread(rgb_path)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def get_depth(path):
        root, idx = os.path.split(path)
        depth_path = os.path.join(root, OmniUtil.DISTANCE_TEMPLATE.format(idx))
        depth = np.load(depth_path)
        return depth
    
    @staticmethod
    def get_seg_data(path):
        root, idx = os.path.split(path)
        seg_labels_path = os.path.join(root, OmniUtil.MASK_LABELS_TEMPLATE.format(idx))
        with open(seg_labels_path, "r") as f:
            seg_labels = json.load(f)
        seg_path = os.path.join(root, OmniUtil.MASK_TEMPLATE.format(idx))
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        return seg, seg_labels
    
    @staticmethod
    def get_single_seg(path, object_name):
        root, idx = os.path.split(path)
        seg_labels_path = os.path.join(root, OmniUtil.MASK_LABELS_TEMPLATE.format(idx))
        with open(seg_labels_path, "r") as f:
            seg_labels = json.load(f)
        seg_path = os.path.join(root, OmniUtil.MASK_TEMPLATE.format(idx))
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        object_mask = (
                seg == OmniUtil.get_mask_rgba(seg_labels, object_name)
            ).all(axis=2)
        return object_mask
        
        
    @staticmethod
    def get_mask_rgba(mask_labels, mask_name):
        name_list = [name_dict["class"] for name_dict in list(mask_labels.values())]
        if mask_name not in name_list:
            return None
        rgba_list = list(mask_labels.keys())
        mask_rgba_str = rgba_list[name_list.index(mask_name)]
        r, g, b, a = re.findall("\d+", mask_rgba_str)
        r, g, b, a = int(b), int(g), int(r), int(a)
        return r, g, b, a
    
    @staticmethod
    def get_rgb_feat(path):
        root, idx = os.path.split(path)
        rgb_feat_path = os.path.join(root, OmniUtil.RGB_FEAT_TEMPLATE.format(idx))
        rgb_feat = np.load(rgb_feat_path)
        return rgb_feat
    
    @staticmethod
    def get_target_object_list(path):
        return OmniUtil.get_object_list(path, contains_non_obj=False) # TODO: generalize this
        

    @staticmethod
    def get_transform_mat(path):
        root, idx = os.path.split(path)
        camera_params_path = os.path.join(
            root, OmniUtil.CAMERA_PARAMS_TEMPLATE.format(idx)
        )
        with open(camera_params_path, "r") as f:
            raw_camera_params = json.load(f)
        cam_transform = np.asarray(raw_camera_params["cameraViewTransform"]).reshape(
            (4, 4)
        )
        real_cam_transform = np.linalg.inv(cam_transform).T
        real_cam_transform = real_cam_transform.dot(
            np.mat([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )
        return real_cam_transform

    @staticmethod
    def get_intrinsic_matrix(path):
        root, idx = os.path.split(path)
        camera_params_path = os.path.join(
            root, OmniUtil.CAMERA_PARAMS_TEMPLATE.format(idx)
        )
        with open(camera_params_path, "r") as f:
            raw_camera_params = json.load(f)
        h_aperture = raw_camera_params["cameraAperture"][0]
        v_aperture = raw_camera_params["cameraAperture"][1]
        focal_length = raw_camera_params["cameraFocalLength"]
        h_resolution = raw_camera_params["renderProductResolution"][0]
        v_resolution = raw_camera_params["renderProductResolution"][1]
        focal_x = h_resolution * focal_length / h_aperture
        focal_y = v_resolution * focal_length / v_aperture
        center_x = h_resolution / 2
        center_y = v_resolution / 2
        intrinsic_matrix = np.array(
            [
                [focal_x, 0, center_x],
                [0, focal_y, center_y],
                [0, 0, 1],
            ]
        )
        return intrinsic_matrix
    
    @staticmethod
    def get_extrinsic_matrix(path):
        root, idx = os.path.split(path)
        camera_params_path = os.path.join(
            root, OmniUtil.CAMERA_PARAMS_TEMPLATE.format(idx)
        )
        with open(camera_params_path, "r") as f:
            raw_camera_params = json.load(f)
        cam_transform = np.asarray(raw_camera_params["cameraViewTransform"]).reshape(
            (4, 4)
        )
        real_cam_transform = np.linalg.inv(cam_transform).T
        real_cam_transform = real_cam_transform.dot(
            np.mat([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )
        return real_cam_transform
    
    @staticmethod
    def get_scene_data(path):
        root, _ = os.path.split(path)
        scene_data_path = os.path.join(
            root, "scene.pickle"
        )
        with open(scene_data_path, "rb") as f:
            scene_data = pickle.load(f)
        return scene_data
    
    @staticmethod
    def get_o2c_pose(path, object_name):
        scene_data = OmniUtil.get_scene_data(path)
        cam_pose = OmniUtil.get_extrinsic_matrix(path)
        pos = scene_data[object_name]["position"]
        quat = scene_data[object_name]["rotation"]
        rot = R.from_quat(quat).as_matrix()
        obj_pose = np.eye(4)
        obj_pose[:3, :3] = rot
        obj_pose[:3, 3] = pos
        obj_cam_pose = np.linalg.inv(cam_pose) @ obj_pose
        return np.asarray(obj_cam_pose)

if __name__ == "__main__":
    test_path = r"/mnt/h/AI/Datasets/nbv1/sample_one/scene_0/0050"
    obj_list = OmniUtil.get_object_list(test_path, contains_non_obj=True)
    print(obj_list)
    pts = OmniUtil.get_segmented_points(test_path, target_list=obj_list)
    np.savetxt("pts1.txt", pts)
