import rospy

from controllers import *
from model import *
from robot_sim import *
from utils import *


gui = True
dt = 1.0 / 60.0

rospy.init_node("demo")

env = SimPandaEnv(gui)
model = Model("panda_link0", "panda_ee")

q, dq = env.arm.get_state()
x0 = model.pose(q)

controller = CartesianPoseController(model, x0)

marker = InteractiveMarkerWrapper("target", "panda_link0", x0)

while True:
    controller.set_target(marker.pose)
    q, dq = env.arm.get_state()
    cmd = controller.update(q, dq)
    env.arm.set_desired_joint_velocities(cmd)

    env.forward(dt)
