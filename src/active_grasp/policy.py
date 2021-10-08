import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy

from .timer import Timer
from .rviz import Visualizer
from robot_helpers.model import KDLModel
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import *
from vgn.perception import UniformTSDFVolume


class Policy:
    def __init__(self):
        self.load_parameters()
        self.init_robot_model()
        self.init_visualizer()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.T_grasp_ee = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.task_frame = "task"
        info_topic = rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        self.qual_threshold = rospy.get_param("vgn/qual_threshold")

    def init_robot_model(self):
        self.model = KDLModel.from_parameter_server(self.base_frame, self.cam_frame)
        self.ee_model = KDLModel.from_parameter_server(self.base_frame, "panda_link8")

    def init_visualizer(self):
        self.vis = Visualizer()

    def is_feasible(self, view, q_init=None):
        if q_init is None:
            q_init = [0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79]
        return self.model.ik(q_init, view) is not None

    def activate(self, bbox, view_sphere):
        self.vis.clear()

        self.bbox = bbox
        self.view_sphere = view_sphere
        self.vis.bbox(self.base_frame, self.bbox)

        self.calibrate_task_frame()

        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.info = {}

    def calibrate_task_frame(self):
        self.T_base_task = Transform.translation(self.bbox.center - np.full(3, 0.15))
        self.T_task_base = self.T_base_task.inv()
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(1.0)  # Wait for tf tree to be updated

    def update(self, img, x, q):
        raise NotImplementedError

    def sort_grasps(self, grasps, qualities, q):
        """
        1. Transform grasp configurations into base_frame
        2. Check whether the finger tips lie within the bounding box
        3. Remove grasps for which no IK solution was found
        4. Sort grasps according to score_fn
        """
        filtered_grasps, scores = [], []
        for grasp, quality in zip(grasps, qualities):
            pose = self.T_base_task * grasp.pose
            R, t = pose.rotation, pose.translation
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if self.bbox.is_inside(tip):
                grasp.pose = pose
                q_grasp = self.ee_model.ik(q, pose * self.T_grasp_ee)
                if q_grasp is not None:
                    filtered_grasps.append(grasp)
                    scores.append(self.score_fn(grasp, quality, q, q_grasp))
        filtered_grasps, scores = np.asarray(filtered_grasps), np.asarray(scores)
        i = np.argsort(-scores)
        return filtered_grasps[i], qualities[i], scores[i]

    def score_fn(self, grasp, quality, q, q_grasp):
        return -np.linalg.norm(q - q_grasp)


class SingleViewPolicy(Policy):
    def update(self, img, x, q):
        linear, _ = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.views.append(x)
            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
            tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
            self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
            self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())

            out = self.vgn.predict(tsdf_grid)
            self.vis.quality(self.task_frame, voxel_size, out.qual, 0.5)

            grasps, qualities = select_grid(voxel_size, out, self.qual_threshold)
            grasps, _ = self.sort_grasps(grasps, qualities, q)

            if len(grasps) > 0:
                self.best_grasp = grasps[0]
                self.vis.grasps(self.base_frame, grasps)
                self.vis.best_grasp(self.base_frame, self.best_grasp)

            self.done = True


class MultiViewPolicy(Policy):
    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)
        self.T = 10  # Window size of grasp prediction history
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)

    def integrate(self, img, x, q):
        self.views.append(x)
        self.vis.path(self.base_frame, self.views)

        with Timer("tsdf_integration"):
            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
        self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
        self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())

        with Timer("grasp_prediction"):
            tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
            out = self.vgn.predict(tsdf_grid)
        self.vis.quality(self.task_frame, self.tsdf.voxel_size, out.qual, 0.9)

        t = (len(self.views) - 1) % self.T
        self.qual_hist[t, ...] = out.qual

        with Timer("grasp_selection"):
            grasps, qualities = select_grid(voxel_size, out, self.qual_threshold)
            grasps, qualities, _ = self.sort_grasps(grasps, qualities, q)

        if len(grasps) > 0:
            self.best_grasp = grasps[0]
            # self.vis.grasps(self.base_frame, grasps, qualities)
            self.vis.grasp(self.base_frame, self.best_grasp, qualities[0])
        else:
            self.best_grasp = None
            self.vis.clear_grasp()


def compute_error(x_d, x):
    linear = x_d.translation - x.translation
    angular = (x_d.rotation * x.rotation.inv()).as_rotvec()
    return linear, angular


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id, *args, **kwargs):
    if id in registry:
        return registry[id](*args, **kwargs)
    else:
        raise ValueError("{} policy does not exist.".format(id))
