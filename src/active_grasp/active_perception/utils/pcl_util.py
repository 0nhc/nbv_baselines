import numpy as np
import torch
from scipy.spatial.distance import cdist


class PclUtil:
    CHAMFER = 1

    @staticmethod
    def transform(pts, pose=np.eye(4), scale=np.ones(3), inverse=False):
        aug_scale = np.ones(4)
        aug_scale[:3] = scale
        aug_scale_mat = np.diag(aug_scale)
        scale_pose = pose @ aug_scale_mat
        aug_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
        if inverse:
            scale_pose = np.linalg.inv(scale_pose)
        transformed_pts = scale_pose @ aug_pts.T
        return transformed_pts.T[:, :3]
    
    @staticmethod
    def cam2canonical(cam_pts, cam2canonical_pose):
        aug_pts = np.hstack((cam_pts, np.ones((cam_pts.shape[0], 1))))
        transformed_pts = cam2canonical_pose @ aug_pts.T
        return transformed_pts.T[:, :3]

    @staticmethod
    def transform_batch(pts, pose, scale, inverse=False):
        batch_size = pts.shape[0]
        aug_scale_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(3):
            aug_scale_mat[..., i, i] = scale[..., i]
        scale_pose = pose @ aug_scale_mat
        aug_pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
        if inverse:
            scale_pose = torch.inverse(scale_pose)
        transformers_pts = scale_pose @ aug_pts.transpose(1, 2)
        return transformers_pts.transpose(1, 2)[..., :3]

    @staticmethod
    def transform_n_batch(pts, pose, scale=None, inverse=False):
        transformed_pts_shape = (pts.shape[0], pose.shape[1], pts.shape[1], pts.shape[2])
        transformed_pts = np.zeros(transformed_pts_shape)
        batch_size = pose.shape[0]
        n = pose.shape[1]
        if scale is None:
            scale = np.ones((batch_size, n, 3))
        for batch_i in range(batch_size):
            for i in range(n):
                transformed_pts[batch_i, i, :, :] = PclUtil.transform(pts[batch_i], pose[batch_i, i],
                                                                      scale[batch_i, i], inverse)
        return transformed_pts

    @staticmethod
    def chamfer_distance(pts1, pts2):
        dist_matrix1 = cdist(pts1, pts2, 'euclidean')
        dist_matrix2 = cdist(pts2, pts1, 'euclidean')
        chamfer_dist = np.mean(np.min(dist_matrix1, axis=1)) + np.mean(np.min(dist_matrix2, axis=1))
        return chamfer_dist

    @staticmethod
    def distance(pts1, pts2, eval_type=1):
        if eval_type == PclUtil.CHAMFER:
            return PclUtil.chamfer_distance(pts1, pts2)
        else:
            raise ValueError('Unknown evaluation type:', eval_type)

    @staticmethod
    def sample_pcl(pcl, n_pts=1024):
        indices = np.random.choice(pcl.shape[0], n_pts, replace=pcl.shape[0] < n_pts)
        return pcl[indices, :]


if __name__ == '__main__':
    batch_pts = np.random.random((2, 16, 3))
    batch_n_pose = np.random.random((2, 3, 4, 4))
    batch_n_scale = np.random.random((2, 3, 3))
    poses = PclUtil.transform_n_batch(batch_pts, batch_n_pose, batch_n_scale)
