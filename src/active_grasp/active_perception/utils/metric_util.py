import numpy as np


class MetricUtil:

    @staticmethod
    def rotate_around(axis, angle_deg):
        angle = angle_deg * np.pi / 180
        if axis == "x":
            return np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        elif axis == "y":
            return np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == "z":
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])
        else:
            raise ValueError("Invalid axis")

    @staticmethod
    def basic_rot_diff(r0, r1):
        mat_diff = np.matmul(r0, r1.swapaxes(-1, -2))
        diff = np.trace(mat_diff) - 1
        return np.arccos(np.clip(diff / 2.0, a_min=-1.0, a_max=1.0))

    @staticmethod
    def axis_rot_diff(r0, r1, axis):
        axis1, axis2 = r0[..., axis], r1[..., axis]
        diff = np.sum(axis1 * axis2, axis=-1)
        return np.arccos(np.clip(diff, a_min=-1.0, a_max=1.0))

    @staticmethod
    def turn_rot_diff(r0, r1, axis, turn_degrees):
        diffs = []
        for i in turn_degrees:
            rotation_matrix = MetricUtil.rotate_around(axis, i)
            diffs.append(MetricUtil.basic_rot_diff(np.matmul(r0, rotation_matrix), r1))
        return np.min(diffs, axis=0)

    @staticmethod
    def rot_diff_rad(r0, r1, sym):

        axis_map = {0: "x", 1: "y", 2: "z"}
        if sym is None or sym == 0:  # no symmetry
            return MetricUtil.basic_rot_diff(r0, r1)
        elif sym in [1, 2, 3]:  # free rotation around axis
            return MetricUtil.axis_rot_diff(r0, r1, sym - 1)
        else:  # symmetry
            turns = 0
            axis_idx = 0
            if sym in [4, 5, 6]:  # half turn
                axis_idx = sym - 4
                turns = 2
            elif sym in [7, 8, 9]:  # quarter turn
                axis_idx = sym - 7
                turns = 4
            turn_degrees = np.arange(0, 360, 360 / turns)
            return MetricUtil.turn_rot_diff(r0, r1, axis_map[axis_idx], turn_degrees)

    @staticmethod
    def collect_metric(pred_pose_mat, gt_pose_mat, sym):
        pred_rot_mat = pred_pose_mat[:, :3, :3]
        gt_rot_mat = gt_pose_mat[:, :3, :3]
        pred_trans = pred_pose_mat[:, :3, 3]
        gt_trans = gt_pose_mat[:, :3, 3]

        trans_error = []
        rot_error = []
        for i in range(pred_rot_mat.shape[0]):
            tdiff = np.linalg.norm(pred_trans[i] - gt_trans[i], ord=2) * 100
            rdiff = MetricUtil.rot_diff_rad(pred_rot_mat[i], gt_rot_mat[i], sym[i]) / np.pi * 180.0
            trans_error.append(tdiff)
            rot_error.append(rdiff)

        rot_error = {
            'mean': np.mean(rot_error),
            'median': np.median(rot_error),
            'item': rot_error,
        }
        trans_error = {
            'mean': np.mean(trans_error),
            'median': np.median(trans_error),
            'item': trans_error,
        }
        error = {'rot_error': rot_error,
                 'trans_error': trans_error}
        return error


# -------------- Debug ---------------

def test_MetricUtil():
    print("test case 0: no rotation")
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 0) * 180 / np.pi)
    print("test case 1: 29 degree rotation around x-axis")
    rotation_matrix = MetricUtil.rotate_around("x", 29)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 0) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 1) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 8) * 180 / np.pi)
    print("test case 2: 90 degree rotation around y-axis")
    rotation_matrix = MetricUtil.rotate_around("y", 90)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 0) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 2) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 8) * 180 / np.pi)
    print("test case 3: 60 degree rotation around y-axis")
    rotation_matrix = MetricUtil.rotate_around("y", 60)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 0) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 2) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 8) * 180 / np.pi)
    print("test case 4: 78 degree rotation around z-axis and 60 degree rotation around x-axis")
    rotation_matrix = MetricUtil.rotate_around("z", 78) @ MetricUtil.rotate_around("x", 60)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 0) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 2) * 180 / np.pi)
    print(MetricUtil.rot_diff_rad(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), rotation_matrix, 8) * 180 / np.pi)


if __name__ == "__main__":
    pass
    test_MetricUtil()
