import json
import numpy as np
import requests
import torch
from PIL import Image

from utils.cache_util import LRUCache


class ViewUtil:
    view_cache = LRUCache(1024)
    def load_camera_pose_from_frame(camera_params_path):
        with open(camera_params_path, "r") as f:
            camera_params = json.load(f)
        
        view_transform = camera_params["cameraViewTransform"]
        view_transform = np.resize(view_transform, (4,4))
        view_transform = np.linalg.inv(view_transform).T
        offset = np.mat([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        view_transform = view_transform.dot(offset)
        return view_transform

    def save_image(rgb, filename):
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb, 'RGB')
        img.save(filename)

    def save_depth(depth, filename):
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        depth_img = Image.fromarray(depth)
        depth_img.save(filename)

    def save_segmentation(seg, filename):
        if seg.dtype != np.uint8:
            seg = seg.astype(np.uint8)
        seg_img = Image.fromarray(seg)
        seg_img.save(filename)
    
    @staticmethod
    def get_view(camera_pose,source, data_type,scene,port):
        camera_pose_tuple = tuple(map(tuple, camera_pose.tolist()))
        cache_key = (camera_pose_tuple, source, data_type, scene, port)
        cached_result = ViewUtil.view_cache.get(cache_key)
        if cached_result:
            print("Cache hit")
            return cached_result
        
        url = f"http://127.0.0.1:{port}/get_images"
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'camera_pose': camera_pose.tolist(),
            'data_type': data_type,
            'source': source,
            'scene': scene
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            results = response.json()
            
            rgb = np.asarray(results['rgb'],dtype=np.uint8)
            depth = np.asarray(results['depth'])/1000
            seg = np.asarray(results['segmentation'])
            seg_labels = results['segmentation_labels']
            camera_params = results['camera_params']
            ViewUtil.view_cache.put(cache_key, (rgb, depth, seg, seg_labels, camera_params))
            return rgb, depth, seg, seg_labels, camera_params
        else:
            return None
        
    @staticmethod
    def get_object_pose_batch(K, mesh, rgb_batch, depth_batch, mask_batch, gt_pose_batch ,port):
        url = f"http://127.0.0.1:{port}/predict_estimation_batch"
        headers = {
            'Content-Type': 'application/json'
        }
        mesh_data = {
            'vertices': mesh.vertices.tolist(),
            'faces': mesh.faces.tolist()
        }
        data = {
            'K': K.tolist(),
            'rgb_batch': rgb_batch.tolist(),
            'depth_batch': depth_batch.tolist(),
            'mask_batch': mask_batch.tolist(),
            'mesh': mesh_data,
            'gt_pose_batch': gt_pose_batch.tolist()
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            results = response.json()
            pose_batch = np.array(results['pose_batch'])
            results_batch = results["eval_result_batch"]
            return pose_batch, results_batch
        else:
            return None
        
    @staticmethod
    def get_visualized_result(K, mesh, rgb, pose ,port):
        url = f"http://127.0.0.1:{port}/get_visualized_result"
        headers = {
            'Content-Type': 'application/json'
        }
        mesh_data = {
            'vertices': mesh.vertices.tolist(),
            'faces': mesh.faces.tolist()
        }
        data = {
            'K': K.tolist(),
            'rgb': rgb.tolist(),
            'mesh': mesh_data,
            'pose': pose.tolist()
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            results = response.json()
            vis_rgb = np.array(results['vis_rgb'])
            return vis_rgb
        else:
            return None
        
    @staticmethod
    def get_object_pose(K, mesh, rgb, depth, mask, gt_pose ,port):
        url = f"http://127.0.0.1:{port}/predict_estimation"
        headers = {
            'Content-Type': 'application/json'
        }
        mesh_data = {
            'vertices': mesh.vertices.tolist(),
            'faces': mesh.faces.tolist()
        }
        data = {
            'K': K.tolist(),
            'rgb': rgb.tolist(),
            'depth': depth.tolist(),
            'mask': mask.tolist(),
            'mesh': mesh_data,
            'gt_pose': gt_pose.tolist()
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            results = response.json()
            pose_batch = np.array(results['pose_batch'])
            results_batch = results["eval_result_batch"]
            return pose_batch, results_batch
        else:
            return None
        
    def get_pts_dict(depth, seg, seg_labels, camera_params):
        cx = camera_params['cx']
        cy = camera_params['cy']
        fx = camera_params['fx']
        fy = camera_params['fy']
        width = camera_params['width']
        height = camera_params['height']
        pts_dict = {name: [] for name in seg_labels.values()}
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        labels = seg.reshape(-1)
        for label, name in seg_labels.items():
            mask = labels == int(label)
            pts_dict[name] = points[mask]
        return pts_dict

    def get_object_center_from_pts_dict(obj,pts_dict):
        if obj is None:
            for _, pts in pts_dict.items():
                if pts.size != 0:
                    obj_pts = pts
                    break
        else:
            obj_pts = pts_dict[obj]
            if obj_pts.size == 0:
                for _, pts in pts_dict.items():
                    if pts.size != 0:
                        obj_pts = pts
                        break
        obj_center = obj_pts.mean(axis=0)
        return obj_center
    
    def get_pts_center(pts):
        pts_center = pts.mean(axis=0)
        return pts_center

    def get_scene_pts(pts_dict):
        if any(isinstance(pts, torch.Tensor) for pts in pts_dict.values()):
            scene_pts = torch.cat([pts for _, pts in pts_dict.items()], dim=0)
            return scene_pts
        else:
            scene_pts = np.concatenate([pts for _, pts in pts_dict.items()])
            return scene_pts

    def crop_pts(scene_pts, crop_center, radius=0.2):
        if isinstance(scene_pts, torch.Tensor):
            crop_mask = torch.norm(scene_pts - crop_center, dim=1) < radius
            return scene_pts[crop_mask]
        else:
            crop_mask = np.linalg.norm(scene_pts - crop_center, axis=1) < radius
            return scene_pts[crop_mask]

    def crop_pts_dict(pts_dict, crop_center, radius=0.2, min_pts_num = 5000):
        crop_dict = {}
        max_loop = 100
        loop = 0
        while(loop<=max_loop):
            croped_length = 0
            for obj, pts in pts_dict.items():
                if isinstance(pts, torch.Tensor):
                    crop_mask = torch.norm(pts - crop_center, dim=1) < radius
                    crop_dict[obj] = pts[crop_mask]
                else:
                    crop_mask = np.linalg.norm(pts - crop_center, axis=1) < radius
                    crop_dict[obj] = pts[crop_mask]
                croped_length += crop_dict[obj].shape[0]
            if croped_length >= min_pts_num:
                break
            radius += 0.02
            loop += 1
        return crop_dict
    
    def get_cam_pose_focused_on_point(point_w, cam_pose_w, old_camera_center_w):
        distance = np.linalg.norm(point_w-old_camera_center_w)
        z_axis_camera = cam_pose_w[:3, 2].reshape(-1)
        new_camera_position_w = point_w - distance * z_axis_camera
        new_camera_pose_w = cam_pose_w.copy()
        new_camera_pose_w[:3, 3] = new_camera_position_w.reshape((3,1))
        return new_camera_pose_w