import numpy as np

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform
import vgn.rviz
from vgn.utils import *


cm = lambda s: tuple([float(1 - s), float(s), float(0)])
red = [1.0, 0.0, 0.0]
blue = [0, 0.6, 1.0]
grey = [0.9, 0.9, 0.9]

def create_grasp_marker(frame, grasp, color, ns, id=0, depth=0.05, radius=0.005):
    # Faster grasp marker using Marker.LINE_LIST
    pose, w, d, scale = grasp.pose, grasp.width, depth, [radius, 0.0, 0.0]
    points = [[0, -w / 2, d], [0, -w / 2, 0], [0, w / 2, 0], [0, w / 2, d]]
    return create_line_strip_marker(frame, pose, scale, color, points, ns, id)

class Visualizer(vgn.rviz.Visualizer):
    def clear_ig_views(self):
        markers = [Marker(action=Marker.DELETE, ns="ig_views", id=i) for i in range(24)]
        self.draw(markers)

    def clear_grasps(self):
        markers = [Marker(action=Marker.DELETE, ns="grasps", id=i) for i in range(self.num_grasps)]
        self.draw(markers)
        self.num_grasps = 0

    def grasps(self, frame, grasps, qualities, vmin=0.5, vmax=1.0):
        markers = []
        self.num_grasps = 0
        for i, (grasp, quality) in enumerate(zip(grasps, qualities)):
            color = cm((quality - vmin) / (vmax - vmin))
            markers.append(create_grasp_marker(frame, grasp, color, "grasps", i))
            self.num_grasps += 1
        self.draw(markers)

    def bbox(self, frame, bbox):
        pose = Transform.identity()
        scale = [0.004, 0.0, 0.0]
        color = red
        lines = box_lines(bbox.min, bbox.max)
        marker = create_line_list_marker(frame, pose, scale, color, lines, "bbox")
        self.draw([marker])

    def ig_views(self, frame, intrinsic, views, values):
        vmin, vmax = min(values), max(values)
        scale = [0.002, 0.0, 0.0]
        near, far = 0.0, 0.02
        markers = []
        for i, (view, value) in enumerate(zip(views, values)):
            color = cm((value - vmin) / (vmax - vmin))
            marker = create_view_marker(
                frame,
                view,
                scale,
                color,
                intrinsic,
                near,
                far,
                ns="ig_views",
                id=i,
            )
            markers.append(marker)
        self.draw(markers)

    def path(self, frame, intrinsic, views):
        markers = []
        points = [p.translation for p in views]

        spheres = create_sphere_list_marker(
            frame,
            Transform.identity(),
            np.full(3, 0.008),
            blue,
            points,
            "path",
            0,
        )
        markers.append(spheres)

        if len(views) > 1:
            lines = create_line_strip_marker(
                frame,
                Transform.identity(),
                [0.002, 0.0, 0.0],
                blue,
                points,
                "path",
                1,
            )
            markers.append(lines)

        for i, view in enumerate(views[::4]):
            markers.append(
                create_view_marker(
                    frame,
                    view,
                    [0.002, 0.0, 0.0],
                    blue,
                    intrinsic,
                    0.0,
                    0.02,
                    ns="views",
                    id=i,
                )
            )

        self.draw(markers)

    def point(self, frame, position):
        marker = create_sphere_marker(
            frame,
            Transform.from_translation(position),
            np.full(3, 0.01),
            [0, 0, 1],
            "point",
        )
        self.draw([marker])

    def rays(self, frame, origin, directions, t_max=1.0):
        lines = [[origin, origin + t_max * direction] for direction in directions]
        marker = create_line_list_marker(
            frame,
            Transform.identity(),
            [0.001, 0.0, 0.0],
            grey,
            lines,
            "rays",
        )
        self.draw([marker])


def create_view_marker(frame, pose, scale, color, intrinsic, near, far, ns="", id=0):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    x_n = near * intrinsic.width / (2.0 * intrinsic.fx)
    y_n = near * intrinsic.height / (2.0 * intrinsic.fy)
    z_n = near
    x_f = far * intrinsic.width / (2.0 * intrinsic.fx)
    y_f = far * intrinsic.height / (2.0 * intrinsic.fy)
    z_f = far
    points = [
        [x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, -y_n, z_n],
        [-x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, -y_f, z_f],
        [-x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, y_f, z_f],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_n, y_n, z_n],
        [-x_f, y_f, z_f],
        [-x_n, -y_n, z_n],
        [-x_f, -y_f, z_f],
        [x_n, -y_n, z_n],
        [x_f, -y_f, z_f],
    ]
    marker.points = [to_point_msg(p) for p in points]
    return marker
