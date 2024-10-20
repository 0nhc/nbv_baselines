from controller_manager_msgs.srv import *
import copy
import cv_bridge
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import Image
import trimesh

from .bbox import from_bbox_msg
from .timer import Timer
from active_grasp.srv import *
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from robot_helpers.ros.panda import PandaArmClient, PandaGripperClient
from robot_helpers.ros.moveit import MoveItClient, create_collision_object_from_mesh
from robot_helpers.spatial import Rotation, Transform
from vgn.utils import look_at, cartesian_to_spherical, spherical_to_cartesian


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.load_parameters()
        self.init_service_proxies()
        self.init_robot_connection()
        self.init_moveit()
        self.init_camera_stream()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.T_grasp_ee = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.depth_topic = rospy.get_param("~camera/depth_topic")
        self.seg_topic = rospy.get_param("~camera/seg_topic")
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.control_rate = rospy.get_param("~control_rate")
        self.linear_vel = rospy.get_param("~linear_vel")
        self.move_to_target_threshold = rospy.get_param("~move_to_target_threshold")
        self.policy_rate = rospy.get_param("policy/rate")

    def init_service_proxies(self):
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.get_target_id = rospy.ServiceProxy("get_target_seg_id", TargetID)

    def init_robot_connection(self):
        self.arm = PandaArmClient()
        self.gripper = PandaGripperClient()
        topic = rospy.get_param("cartesian_velocity_controller/topic")
        self.cartesian_vel_pub = rospy.Publisher(topic, Twist, queue_size=10)

    def init_moveit(self):
        self.moveit = MoveItClient("panda_arm")
        rospy.sleep(1.0)  # Wait for connections to be established.
        self.moveit.move_group.set_planner_id("RRTstarkConfigDefault")
        self.moveit.move_group.set_planning_time(3.0)

    def switch_to_cartesian_velocity_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["cartesian_velocity_controller"]
        req.stop_controllers = ["position_joint_trajectory_controller"]
        req.strictness = 1
        self.switch_controller(req)

    def switch_to_joint_trajectory_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        req.strictness = 1
        self.switch_controller(req)

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(self.seg_topic, Image, self.seg_cb, queue_size=1)

    def depth_cb(self, msg):
        self.latest_depth_msg = msg
    
    def seg_cb(self, msg):
        self.latest_seg_msg = msg

    def run(self):
        bbox = self.reset()
        self.switch_to_cartesian_velocity_control()
        with Timer("search_time"):
            grasp = self.search_grasp(bbox)
        if grasp:
            self.switch_to_joint_trajectory_control()
            with Timer("grasp_time"):
                res = self.execute_grasp(grasp)
        else:
            res = "aborted"
        return self.collect_info(res)

    def reset(self):
        Timer.reset()
        self.moveit.scene.clear()
        res = self.reset_env(ResetRequest())
        rospy.sleep(1.0)  # Wait for the TF tree to be updated.
        return from_bbox_msg(res.bbox)

    def search_grasp(self, bbox):
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.policy.activate(bbox, self.view_sphere)
        timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.send_vel_cmd)
        r = rospy.Rate(self.control_rate)

        if(self.policy.policy_type=="single_view"):
            while not self.policy.done:
                depth_img, seg_image, pose, q = self.get_state()
                target_seg_id = self.get_target_id(TargetIDRequest()).id
                # sleep 1s
                for i in range(self.control_rate*1):
                    r.sleep()
                self.policy.update(depth_img, seg_image, target_seg_id, pose, q)
                # Wait for the robot to move to its desired camera pose
                moving_to_The_target = True
                while(moving_to_The_target):
                    depth_img, seg_image, pose, q = self.get_state()
                    current_p = pose.as_matrix()[:3,3]
                    target_p = self.policy.x_d.as_matrix()[:3,3]
                    linear_d = np.sqrt((current_p[0]-target_p[0])**2+
                                    (current_p[1]-target_p[1])**2+
                                    (current_p[2]-target_p[2])**2)
                    if(linear_d < self.move_to_target_threshold):
                        # Arrived
                        moving_to_The_target = False
                    r.sleep()
        elif(self.policy.policy_type=="multi_view"):
            while not self.policy.done:
                depth_img, seg_image, pose, q = self.get_state()
                target_seg_id = self.get_target_id(TargetIDRequest()).id
                self.policy.update(depth_img, seg_image, target_seg_id, pose, q)
                r.sleep()
        else:
            print("Unsupported policy type: "+str(self.policy.policy_type))

        # Wait for a zero command to be sent to the robot.
        rospy.sleep(0.2)
        self.policy.deactivate()
        timer.shutdown()
        return self.policy.best_grasp

    def get_state(self):
        q, _ = self.arm.get_state()
        msg = copy.deepcopy(self.latest_depth_msg)
        depth_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        msg = copy.deepcopy(self.latest_seg_msg)
        seg_img = self.cv_bridge.imgmsg_to_cv2(msg)
        pose = tf.lookup(self.base_frame, self.cam_frame, msg.header.stamp)
        return depth_img, seg_img, pose, q

    def send_vel_cmd(self, event):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            x = tf.lookup(self.base_frame, self.cam_frame)
            cmd = self.compute_velocity_cmd(self.policy.x_d, x)
        self.cartesian_vel_pub.publish(to_twist_msg(cmd))

    def compute_velocity_cmd(self, x_d, x):
        r, theta, phi = cartesian_to_spherical(x.translation - self.view_sphere.center)
        e_t = x_d.translation - x.translation
        e_n = (x.translation - self.view_sphere.center) * (self.view_sphere.r - r) / r
        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n
        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, self.linear_vel) / scale
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv()
        angular = 1.0 * angular.as_rotvec()
        return np.r_[linear, angular]

    def execute_grasp(self, grasp):
        self.create_collision_scene()
        T_base_grasp = self.postprocess(grasp.pose)
        self.gripper.move(0.08)
        T_base_approach = T_base_grasp * Transform.t_[0, 0, -0.06] * self.T_grasp_ee
        success, plan = self.moveit.plan(T_base_approach, 0.2, 0.2)
        if success:
            self.moveit.scene.clear()
            self.moveit.execute(plan)
            rospy.sleep(0.5)  # Wait for the planning scene to be updated
            self.moveit.gotoL(T_base_grasp * self.T_grasp_ee)
            rospy.sleep(0.5)
            self.gripper.grasp()
            T_base_retreat = Transform.t_[0, 0, 0.05] * T_base_grasp * self.T_grasp_ee
            self.moveit.gotoL(T_base_retreat)
            rospy.sleep(1.0)  # Wait to see whether the object slides out of the hand
            success = self.gripper.read() > 0.002
            return "succeeded" if success else "failed"
        else:
            return "no_motion_plan_found"

    def create_collision_scene(self):
        # Segment support surface
        cloud = self.policy.tsdf.get_scene_cloud()
        cloud = cloud.transform(self.policy.T_base_task.as_matrix())
        _, inliers = cloud.segment_plane(0.01, 3, 1000)
        support_cloud = cloud.select_by_index(inliers)
        cloud = cloud.select_by_index(inliers, invert=True)
        # o3d.io.write_point_cloud(f"{time.time():.0f}.pcd", cloud)

        # Add collision object for the support
        self.add_collision_mesh("support", compute_convex_hull(support_cloud))

        # Cluster cloud
        labels = np.array(cloud.cluster_dbscan(eps=0.01, min_points=8))

        # Generate convex collision objects for each segment
        self.hulls = []
        for label in range(labels.max() + 1):
            segment = cloud.select_by_index(np.flatnonzero(labels == label))
            try:
                hull = compute_convex_hull(segment)
                name = f"object_{label}"
                self.add_collision_mesh(name, hull)
                self.hulls.append(hull)
            except:
                # Qhull fails in some edge cases
                pass

    def add_collision_mesh(self, name, mesh):
        frame, pose = self.base_frame, Transform.identity()
        co = create_collision_object_from_mesh(name, frame, pose, mesh)
        self.moveit.scene.add_object(co)

    def postprocess(self, T_base_grasp):
        rot = T_base_grasp.rotation
        if rot.as_matrix()[:, 0][0] < 0:  # Ensure that the camera is pointing forward
            T_base_grasp.rotation = rot * Rotation.from_euler("z", np.pi)
        T_base_grasp *= Transform.t_[0.0, 0.0, 0.01]
        return T_base_grasp

    def collect_info(self, result):
        points = [p.translation for p in self.policy.views]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])
        info = {
            "result": result,
            "view_count": len(points),
            "distance": d,
        }
        info.update(self.policy.info)
        info.update(Timer.timers)
        return info


def compute_convex_hull(cloud):
    hull, _ = cloud.compute_convex_hull()
    triangles, vertices = np.asarray(hull.triangles), np.asarray(hull.vertices)
    return trimesh.base.Trimesh(vertices, triangles)


class ViewHalfSphere:
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError
