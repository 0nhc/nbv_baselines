# Roadmap
* Debug with model output pose
* Change simulation scenes
* Interface with GS-Net

# Updated installation steps fo my PC environment

```sh
# Install Active Grasp
sudo apt install liborocos-kdl-dev
mkdir -p ws/src && cd ws/src
git clone https://github.com/0nhc/active_grasp.git
conda create -n active_grasp python=3.8
cd active_grasp && conda activate active_grasp
pip install -r requirements.txt
conda install libffi==3.3
conda install conda-forge::python-orocos-kdl
cd ..
git clone https://github.com/0nhc/vgn.git -b devel
cd vgn
pip install -r requirements.txt
cd ..
git clone https://github.com/0nhc/robot_helpers.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin build

# Install Active Perception
cd <path-to-your-ws>/src/active_grasp/src/active_grasp/active_perception/modules/module_lib/pointnet2_utils/pointnet2
pip install -e .
```

# Updated Features
* Added our baseline: src/active_grasp/active_perception_policy.py
* Added RGB and Segmentation image publishers. The segmentation ID 1 corresponds to the grasping target object.











# Closed-Loop Next-Best-View Planning for Target-Driven Grasping

This repository contains the implementation of our IROS 2022 submission, _"Closed-Loop Next-Best-View Planning for Target-Driven Grasping"_. [[Paper](http://arxiv.org/abs/2207.10543)][[Video](https://youtu.be/67W_VbSsAMQ)]

## Setup

The experiments were conducted with a Franka Emika Panda arm and a Realsense D435 attached to the wrist of the robot. The code was developed and tested on Ubuntu 20.04 with ROS Noetic. It depends on the following external packages:

- [MoveIt](https://github.com/ros-planning/panda_moveit_config)
- [robot_helpers](https://github.com/mbreyer/robot_helpers)
- [TRAC-IK](http://wiki.ros.org/trac_ik)
- [VGN](https://github.com/ethz-asl/vgn/tree/devel)
- franka_ros and realsense2_camera (only required for hardware experiments)

Additional Python dependencies can be installed with

```
pip install -r requirements.txt
```

Run `catkin build active_grasp` to build the package.

Finally, download the [assets folder](https://drive.google.com/file/d/1xJF9Cd82ybCH3nCdXtQRktTr4swDcNFD/view) and extract it inside the repository.

## Experiments

Start a roscore.

```
roscore
```

To run simulation experiments.

```
roslaunch active_grasp env.launch sim:=true
python3 scripts/run.py nbv
```

To run real-world experiments.

```
roslaunch active_grasp hw.launch
roslaunch active_grasp env.launch sim:=false
python3 scripts/run.py nbv --wait-for-input
```
