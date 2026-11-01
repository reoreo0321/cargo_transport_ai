"""Geometry and controller regression checks; no ROS/Gazebo required."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cargo_transport import field
from cargo_transport.controller import PathFollower as Follower
from cargo_transport.simulation import obstacle_contacts


class FieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = field.load_config(ROOT / "configs/factory_field.json")
        cls.routes = {goal: field.plan(cls.config, "A", goal) for goal in ["B", "C"]}

    def test_both_routes_detour_with_full_footprint_clearance(self):
        c = self.config
        radius = c["robot_radius_m"] + c["safety_margin_m"]
        for goal, route in self.routes.items():
            with self.subTest(goal=goal):
                start = c["stations"]["A"]["position_m"]
                end = c["stations"][goal]["position_m"]
                self.assertFalse(field.segment_free(start, end, c, radius))
                self.assertEqual(tuple(route["points"][0]), tuple(start))
                self.assertEqual(tuple(route["points"][-1]), tuple(end))
                self.assertGreater(route["length_m"], math.dist(start, end))
                self.assertTrue(
                    all(
                        field.segment_free(a, b, c, radius)
                        for a, b in zip(route["points"], route["points"][1:])
                    )
                )

    def test_blocked_route_reports_failure(self):
        c = copy.deepcopy(self.config)
        c["obstacles"].append(
            {"id": "sealed", "bounds_m": [2.5, 0, 3, 8], "height_m": 1}
        )
        with self.assertRaisesRegex(ValueError, "No path"):
            field.plan(c, "A", "B")

    def test_follower_reaches_each_goal_without_cutting_obstacles(self):
        for goal, route in self.routes.items():
            with self.subTest(goal=goal):
                follower = Follower(route, self.config["controller"])
                x, y = route["points"][0]
                q = route["points"][1]
                yaw = math.atan2(q[1] - y, q[0] - x)
                min_clearance = float("inf")
                for _ in range(8000):
                    v, w, d = follower.step(x, y, yaw, 0.04)
                    x += v * math.cos(yaw) * 0.04
                    y += v * math.sin(yaw) * 0.04
                    yaw += w * 0.04
                    min_clearance = min(
                        min_clearance,
                        field.clearance(
                            (x, y), self.config, self.config["robot_radius_m"]
                        ),
                    )
                    if follower.goal_latched and v == 0:
                        break
                self.assertTrue(follower.goal_latched)
                self.assertLess(d["goal_distance_m"], 0.15)
                self.assertGreater(min_clearance, 0.15)

    def test_contacts_distinguish_floor_and_obstacles(self):
        def state(a, b):
            return SimpleNamespace(collision1_name=a, collision2_name=b)

        floor = state("cargo_robot::left_wheel::collision", "floor::ground::collision")
        obstacle = state(
            "obstacle_rack_lower::body::collision", "cargo_robot::base_link::collision"
        )
        wall = state(
            "cargo_robot::base_link::collision", "boundary_west::body::collision"
        )
        self.assertEqual(obstacle_contacts([floor]), [])
        self.assertEqual(len(obstacle_contacts([floor, obstacle, wall])), 2)


if __name__ == "__main__":
    unittest.main()
