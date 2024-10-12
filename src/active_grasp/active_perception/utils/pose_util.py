import numpy as np
import torch
import torch.nn.functional as F

class PoseUtil:
    ROTATION = 1
    TRANSLATION = 2
    SCALE = 3

    @staticmethod
    def get_uniform_translation(trans_m_min, trans_m_max, trans_unit, debug=False):
        if isinstance(trans_m_min, list):
            x_min, y_min, z_min = trans_m_min
            x_max, y_max, z_max = trans_m_max
        else:
            x_min, y_min, z_min = trans_m_min, trans_m_min, trans_m_min
            x_max, y_max, z_max = trans_m_max, trans_m_max, trans_m_max

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        translation = np.array([x, y, z])
        if trans_unit == "cm":
            translation = translation / 100
        if debug:
            print("uniform translation:", translation)
        return translation

    @staticmethod
    def get_uniform_rotation(rot_degree_min=0, rot_degree_max=180, debug=False):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        theta = np.random.uniform(rot_degree_min / 180 * np.pi, rot_degree_max / 180 * np.pi)

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        if debug:
            print("uniform rotation:", theta * 180 / np.pi)
        return R

    @staticmethod
    def get_uniform_pose(trans_min, trans_max, rot_min=0, rot_max=180, trans_unit="cm", debug=False):
        translation = PoseUtil.get_uniform_translation(trans_min, trans_max, trans_unit, debug)
        rotation = PoseUtil.get_uniform_rotation(rot_min, rot_max, debug)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose

    @staticmethod
    def get_n_uniform_pose(trans_min, trans_max, rot_min=0, rot_max=180, n=1,
                           trans_unit="cm", fix=None, contain_canonical=True, debug=False):
        if fix == PoseUtil.ROTATION:
            translations = np.zeros((n, 3))
            for i in range(n):
                translations[i] = PoseUtil.get_uniform_translation(trans_min, trans_max, trans_unit, debug)
            if contain_canonical:
                translations[0] = np.zeros(3)
            rotations = PoseUtil.get_uniform_rotation(rot_min, rot_max, debug)
        elif fix == PoseUtil.TRANSLATION:
            rotations = np.zeros((n, 3, 3))
            for i in range(n):
                rotations[i] = PoseUtil.get_uniform_rotation(rot_min, rot_max, debug)
            if contain_canonical:
                rotations[0] = np.eye(3)
            translations = PoseUtil.get_uniform_translation(trans_min, trans_max, trans_unit, debug)
        else:
            translations = np.zeros((n, 3))
            rotations = np.zeros((n, 3, 3))
            for i in range(n):
                translations[i] = PoseUtil.get_uniform_translation(trans_min, trans_max, trans_unit, debug)
            for i in range(n):
                rotations[i] = PoseUtil.get_uniform_rotation(rot_min, rot_max, debug)
            if contain_canonical:
                translations[0] = np.zeros(3)
                rotations[0] = np.eye(3)

        pose = np.eye(4, 4, k=0)[np.newaxis, :].repeat(n, axis=0)
        pose[:, :3, :3] = rotations
        pose[:, :3, 3] = translations

        return pose

    @staticmethod
    def get_n_uniform_pose_batch(trans_min, trans_max, rot_min=0, rot_max=180, n=1, batch_size=1,
                                 trans_unit="cm", fix=None, contain_canonical=False, debug=False):

        batch_poses = []
        for i in range(batch_size):
            pose = PoseUtil.get_n_uniform_pose(trans_min, trans_max, rot_min, rot_max, n,
                                               trans_unit, fix, contain_canonical, debug)
            batch_poses.append(pose)
        pose_batch = np.stack(batch_poses, axis=0)
        return pose_batch

    @staticmethod
    def get_uniform_scale(scale_min, scale_max, debug=False):
        if isinstance(scale_min, list):
            x_min, y_min, z_min = scale_min
            x_max, y_max, z_max = scale_max
        else:
            x_min, y_min, z_min = scale_min, scale_min, scale_min
            x_max, y_max, z_max = scale_max, scale_max, scale_max

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        scale = np.array([x, y, z])
        if debug:
            print("uniform scale:", scale)
        return scale

    @staticmethod
    def normalize_rotation(rotation, rotation_mode):
        if rotation_mode == 'quat_wxyz' or rotation_mode == 'quat_xyzw':
            rotation /= torch.norm(rotation, dim=-1, keepdim=True)
        elif rotation_mode == 'rot_matrix':
            rot_matrix = PoseUtil.rotation_6d_to_matrix_tensor_batch(rotation)
            rotation[:, :3] = rot_matrix[:, 0, :]
            rotation[:, 3:6] = rot_matrix[:, 1, :]
        elif rotation_mode == 'euler_xyz_sx_cx':
            rot_sin_theta = rotation[:, :3]
            rot_cos_theta = rotation[:, 3:6]
            theta = torch.atan2(rot_sin_theta, rot_cos_theta)
            rotation[:, :3] = torch.sin(theta)
            rotation[:, 3:6] = torch.cos(theta)
        elif rotation_mode == 'euler_xyz':
            pass
        else:
            raise NotImplementedError
        return rotation

    @staticmethod
    def get_pose_dim(rot_mode):
        assert rot_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
            f"the rotation mode {rot_mode} is not supported!"

        if rot_mode == 'quat_wxyz' or rot_mode == 'quat_xyzw':
            pose_dim = 4
        elif rot_mode == 'euler_xyz':
            pose_dim = 3
        elif rot_mode == 'euler_xyz_sx_cx' or rot_mode == 'rot_matrix':
            pose_dim = 6
        else:
            raise NotImplementedError
        return pose_dim

    @staticmethod
    def rotation_6d_to_matrix_tensor_batch(d6: torch.Tensor) -> torch.Tensor:

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    @staticmethod
    def matrix_to_rotation_6d_tensor_batch(matrix: torch.Tensor) -> torch.Tensor:
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    @staticmethod
    def rotation_6d_to_matrix_numpy(d6):
        a1, a2 = d6[:3], d6[3:]
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        return np.stack((b1, b2, b3),axis=-2)

    @staticmethod
    def matrix_to_rotation_6d_numpy(matrix):
        return np.copy(matrix[:2, :]).reshape((6,))



''' ------------ Debug ------------ '''

if __name__ == '__main__':
    for _ in range(1):
        PoseUtil.get_uniform_pose(trans_min=[-25, -25, 10], trans_max=[25, 25, 60],
                                  rot_min=0, rot_max=10, debug=True)
        PoseUtil.get_uniform_scale(scale_min=0.25, scale_max=0.30, debug=True)
    PoseUtil.get_n_uniform_pose_batch(trans_min=[-25, -25, 10], trans_max=[25, 25, 60],
                                      rot_min=0, rot_max=10, batch_size=2, n=2, fix=PoseUtil.TRANSLATION, debug=True)
