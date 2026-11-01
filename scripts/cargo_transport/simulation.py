#!/usr/bin/env python3
"""Paused fixed-step runner derived from run_factory_route.py.

The original asynchronous runner is preserved for historical experiment hashes.
Observe, compute/publish command, advance exactly .05 s, repeat. No learning.
"""
import argparse
import bisect
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import select
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque

from . import field
from .controller import PathFollower as Follower
from .physics import angles, slope

ROOT = field.ROOT


def clip_change(current, target, limit, dt):
    return current + max(-limit * dt, min(limit * dt, target - current))


def obstacle_contacts(states):
    pairs = []
    for state in states:
        names = (state.collision1_name, state.collision2_name)
        if any(name.startswith(("obstacle_", "boundary_")) for name in names):
            pairs.append(names)
    return pairs


def shutdown_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute(a, c, route, folder, follower_factory=None, controller_label=None):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from gazebo_msgs.srv import GetEntityState
    from gazebo_msgs.msg import ContactsState
    from geometry_msgs.msg import Twist

    summary = {
        "route": a.route,
        "status": "error",
        "controller": "fixed-parameter pure pursuit",
        "learned_adaptation": False,
        "planned_length_m": route["length_m"],
    }
    if controller_label:
        summary["controller"] = controller_label
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = os.environ.copy()
    env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{port}"
    env["ROS_DOMAIN_ID"] = str(a.domain_id)
    if a.gui:
        env["DISPLAY"] = env.get("DISPLAY") or ":0"
    os.environ["ROS_DOMAIN_ID"] = str(a.domain_id)
    os.environ["GAZEBO_MASTER_URI"] = env["GAZEBO_MASTER_URI"]
    context = Context()
    initialized = False
    node = executor = publisher = None
    server = client_process = stepper = None
    step_s = 0.05
    summary["simulation_step_s"] = step_s
    collision_pairs = set()
    contact_frames = set()
    contact_messages = 0
    contact_with_floor_frames = set()
    rows = []
    interrupted = False

    def on_contact(message):
        nonlocal contact_messages
        contact_messages += 1
        contact_frames.add(message.header.frame_id)
        for state in message.states:
            if any(
                name.startswith("floor::")
                for name in (state.collision1_name, state.collision2_name)
            ):
                contact_with_floor_frames.add(message.header.frame_id)
        collision_pairs.update(obstacle_contacts(message.states))

    try:
        with (folder / "gzserver.log").open("w") as server_log, (
            folder / "gzclient.log"
        ).open("w") as client_log:
            server = subprocess.Popen(
                [
                    "gzserver",
                    "--verbose",
                    "--pause",
                    "--seed",
                    str(a.seed),
                    "-s",
                    "libgazebo_ros_init.so",
                    str(folder / "field.world"),
                ],
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if a.gui:
                client_process = subprocess.Popen(
                    ["gzclient", "--verbose"],
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            rclpy.init(context=context)
            initialized = True
            node = rclpy.create_node("factory_route_baseline", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            service = node.create_client(GetEntityState, "/gazebo/get_entity_state")
            publisher = node.create_publisher(Twist, "/cargo_robot/cmd_vel", 10)
            node.create_subscription(
                ContactsState,
                "/cargo_robot/contacts",
                on_contact,
                QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE),
            )
            deadline = time.monotonic() + 30
            while (
                not service.wait_for_service(timeout_sec=0.25)
                or publisher.get_subscription_count() < 1
            ):
                executor.spin_once(timeout_sec=0.01)
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Gazebo did not start; inspect gzserver.log")
            publisher.publish(Twist())
            time.sleep(0.05)
            stepper_path = getattr(a, "stepper", ROOT / "logs/runtime/gazebo_stepper")
            world_name = (
                ET.parse(folder / "field.world").getroot().find("world").get("name")
            )
            stepper = subprocess.Popen(
                [str(stepper_path), world_name],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=server_log,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if (
                not select.select([stepper.stdout], [], [], 15)[0]
                or stepper.stdout.readline().strip() != "READY"
            ):
                raise RuntimeError("Fixed-step transport helper did not initialize")

            def read(name, frame):
                request = GetEntityState.Request()
                request.name, request.reference_frame = name, frame
                future = service.call_async(request)
                executor.spin_until_future_complete(future, timeout_sec=3)
                if not future.done():
                    future.cancel()
                    raise RuntimeError("State request timed out")
                result = future.result()
                if result is None or not result.success:
                    raise RuntimeError("State request failed")
                return (
                    result.header.stamp.sec + result.header.stamp.nanosec * 1e-9,
                    result.state,
                )

            start_t, _ = read("cargo_robot::base_link", "world")
            follower = (follower_factory or Follower)(route, c["controller"])
            baseline = None
            last_t = None
            last_advanced_wall = time.monotonic()
            last_print = -5
            stopped_since = None
            actual_distance = 0
            previous_position = None
            slip_x, slip_y = deque(), deque()
            timeout_sim = (
                getattr(a, "max_sim_s", None)
                or route["length_m"] / c["controller"]["cruise_speed_mps"] * 4 + 25
            )
            previous_v, previous_w = 0.0, 0.0
            timeout_wall = time.monotonic() + max(300, timeout_sim * 12)
            while True:
                wall_tick = time.monotonic()
                if wall_tick > timeout_wall:
                    raise RuntimeError("Wall-clock timeout")
                if server.poll() is not None:
                    raise RuntimeError("Gazebo server exited")
                if a.gui and client_process.poll() is not None:
                    raise RuntimeError("Gazebo GUI exited; inspect gzclient.log")
                bt, base = read("cargo_robot::base_link", "world")
                ct, cargo = read("cargo_box", "cargo_robot::base_link")
                t = bt - start_t
                dt = step_s if last_t is None else bt - last_t
                if abs(dt - step_s) > 1e-6 or abs(ct - bt) > 1e-6:
                    raise RuntimeError("Fixed-step observation timestamps disagree")
                if dt < 0:
                    raise RuntimeError("Simulation time moved backwards")
                if dt == 0:
                    # State services can briefly return the same simulation tick.
                    # Keep the previous command; do not create duplicate samples
                    # or advance the experiment profile until time moves again.
                    if wall_tick - last_advanced_wall > 2:
                        raise RuntimeError(
                            "Simulation time did not advance for 2 seconds"
                        )
                    time.sleep(0.01)
                    continue
                last_t = bt
                last_advanced_wall = wall_tick
                p = base.pose.position
                roll, pitch, yaw = angles(base.pose.orientation)
                cp = cargo.pose.position
                cr, cpitch, cyaw = angles(cargo.pose.orientation)
                if not all(
                    math.isfinite(v)
                    for v in [bt, ct, p.x, p.y, p.z, yaw, cp.x, cp.y, cp.z]
                ):
                    raise RuntimeError("Non-finite state")
                if baseline is None and t >= c["controller"]["settle_s"]:
                    baseline = (cp.x, cp.y, cp.z)
                if previous_position is not None:
                    actual_distance += math.dist(previous_position, (p.x, p.y))
                previous_position = (p.x, p.y)
                vx = slope(slip_x, ct, cp.x, 0.12)
                vy = slope(slip_y, ct, cp.y, 0.12)
                geometric_clearance = field.clearance(
                    (p.x, p.y), c, c["robot_radius_m"]
                )
                if baseline is None:
                    v, w = 0.0, 0.0
                    diagnostic = {
                        "path_error_m": follower.nearest(p.x, p.y),
                        "progress_m": follower.progress,
                        "remaining_m": follower.lengths[-1] - follower.progress,
                        "goal_distance_m": math.dist((p.x, p.y), route["points"][-1]),
                        "curvature_command_per_m": 0.0,
                        "goal_latched": False,
                    }
                else:
                    if hasattr(follower, "observe"):
                        follower.observe(
                            {
                                "base_stamp_s": bt,
                                "cargo_stamp_s": ct,
                                "base_v_forward_mps": base.twist.linear.x
                                * math.cos(yaw)
                                + base.twist.linear.y * math.sin(yaw),
                                "yaw_rate_rps": base.twist.angular.z,
                                "previous_cmd_v_mps": previous_v,
                                "previous_cmd_yaw_rps": previous_w,
                                "cargo_dx_m": cp.x - baseline[0],
                                "cargo_dy_m": cp.y - baseline[1],
                                "cargo_displacement_m": math.hypot(
                                    cp.x - baseline[0], cp.y - baseline[1]
                                ),
                                "cargo_vx_rel_mps": vx,
                                "cargo_vy_rel_mps": vy,
                                "cargo_roll_rel_rad": cr,
                                "cargo_pitch_rel_rad": cpitch,
                            }
                        )
                    v, w, diagnostic = follower.step(p.x, p.y, yaw, min(dt, 0.15))
                dx = cp.x - baseline[0] if baseline else None
                dy = cp.y - baseline[1] if baseline else None
                displacement = math.hypot(dx, dy) if baseline else None
                speed = math.hypot(base.twist.linear.x, base.twist.linear.y)
                row = {
                    "elapsed_sim_s": t,
                    "base_stamp_s": bt,
                    "cargo_stamp_s": ct,
                    "x_m": p.x,
                    "y_m": p.y,
                    "yaw_rad": yaw,
                    "speed_mps": speed,
                    "base_v_forward_mps": base.twist.linear.x * math.cos(yaw)
                    + base.twist.linear.y * math.sin(yaw),
                    "base_vx_world_mps": base.twist.linear.x,
                    "base_vy_world_mps": base.twist.linear.y,
                    "previous_cmd_v_mps": previous_v,
                    "previous_cmd_yaw_rps": previous_w,
                    "yaw_rate_rps": base.twist.angular.z,
                    "cmd_v_mps": v,
                    "cmd_yaw_rps": w,
                    "base_roll_rad": roll,
                    "base_pitch_rad": pitch,
                    "cargo_x_rel_m": cp.x,
                    "cargo_y_rel_m": cp.y,
                    "cargo_z_rel_m": cp.z,
                    "cargo_dx_m": dx,
                    "cargo_dy_m": dy,
                    "cargo_displacement_m": displacement,
                    "cargo_vx_rel_mps": vx,
                    "cargo_vy_rel_mps": vy,
                    "cargo_roll_rel_rad": cr,
                    "cargo_pitch_rel_rad": cpitch,
                    "cargo_yaw_rel_rad": cyaw,
                    "footprint_clearance_m": geometric_clearance,
                    "obstacle_contact_pairs": len(collision_pairs),
                    "contact_messages": contact_messages,
                    **diagnostic,
                }
                rows.append(row)
                if collision_pairs:
                    summary["status"] = "obstacle_contact"
                    break
                if geometric_clearance < 0.03:
                    summary["status"] = "footprint_clearance_violation"
                    break
                if baseline and cp.z < baseline[2] - 0.06:
                    summary["status"] = "cargo_fell"
                    break
                if baseline and displacement * 1000 > getattr(
                    a, "abort_slip_mm", float("inf")
                ):
                    summary["status"] = "cargo_slip_abort"
                    break
                if max(abs(cr), abs(cpitch)) > math.radians(
                    getattr(a, "abort_tilt_deg", 90)
                ):
                    summary["status"] = "cargo_tilt_abort"
                    break
                if max(abs(roll), abs(pitch)) > math.radians(35):
                    summary["status"] = "robot_tipped"
                    break
                if t > 4 and (len(contact_frames) < 4 or not contact_with_floor_frames):
                    raise RuntimeError("Contact sensors not confirmed active")
                if t > timeout_sim:
                    summary["status"] = "route_timeout"
                    break
                command = Twist()
                command.linear.x, command.angular.z = v, w
                publisher.publish(command)
                previous_v, previous_w = v, w
                if (
                    getattr(follower, "experiment_done", False)
                    and v < 0.001
                    and abs(w) < 0.001
                    and speed < 0.02
                ):
                    stopped_since = t if stopped_since is None else stopped_since
                    if t - stopped_since >= 0.3:
                        summary["status"] = "profile_completed"
                        break
                elif getattr(follower, "experiment_done", False):
                    stopped_since = None
                if follower.goal_latched and v < 0.001 and speed < 0.02:
                    stopped_since = t if stopped_since is None else stopped_since
                    if t - stopped_since >= 0.5:
                        summary["status"] = (
                            "goal_reached"
                            if diagnostic["goal_distance_m"] <= 0.15
                            else "goal_stop_error"
                        )
                        break
                elif not getattr(follower, "experiment_done", False):
                    stopped_since = None
                if t - last_print >= 5:
                    print(
                        f"t={t:.1f}s | progress={follower.progress:.2f}/{follower.lengths[-1]:.2f}m | "
                        f'error={diagnostic["path_error_m"]:.3f}m | '
                        f"slip={(displacement or 0)*1000:.2f}mm",
                        flush=True,
                    )
                    last_print = t
                # Publish while paused; allow the ROS subscriber callback to run
                # before physics advances. State timestamps, not wall time,
                # establish the fixed simulation observation interval.
                time.sleep(0.005)
                stepper.stdin.write("50\n")
                stepper.stdin.flush()
                if (
                    not select.select([stepper.stdout], [], [], 5)[0]
                    or stepper.stdout.readline().strip() != "SENT 50"
                ):
                    raise RuntimeError("Fixed-step command was not sent")
                target_time = bt + step_s
                deadline = time.monotonic() + 10
                while True:
                    advanced, _ = read("cargo_robot::base_link", "world")
                    if advanced >= target_time - 1e-6:
                        if abs(advanced - target_time) > 1e-6:
                            raise RuntimeError(
                                "Physics advanced beyond requested fixed step"
                            )
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("Fixed-step physics did not finish")
                    time.sleep(0.001)
            publisher.publish(Twist())
            for _ in range(5):
                executor.spin_once(timeout_sec=0.01)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        interrupted = True
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = str(error)
    finally:
        if publisher is not None:
            try:
                publisher.publish(Twist())
            except Exception:
                pass
        if executor is not None:
            executor.shutdown(timeout_sec=1)
        if node is not None:
            node.destroy_node()
        if initialized:
            rclpy.shutdown(context=context)
        shutdown_process(client_process)
        shutdown_process(stepper)
        shutdown_process(server)
        if rows:
            with (folder / "samples.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
                )
                writer.writeheader()
                writer.writerows(rows)
            valid = [r for r in rows if r["cargo_displacement_m"] is not None]
            summary.update(
                {
                    "sim_duration_s": rows[-1]["elapsed_sim_s"],
                    "actual_distance_m": actual_distance,
                    "goal_error_m": rows[-1]["goal_distance_m"],
                    "final_speed_mps": rows[-1]["speed_mps"],
                    "max_path_error_m": max(r["path_error_m"] for r in rows),
                    "min_footprint_clearance_m": min(
                        r["footprint_clearance_m"] for r in rows
                    ),
                    "max_cargo_displacement_mm": max(
                        (r["cargo_displacement_m"] * 1000 for r in valid), default=0
                    ),
                    "max_cargo_tilt_deg": max(
                        (
                            math.degrees(
                                math.hypot(
                                    r["cargo_roll_rel_rad"], r["cargo_pitch_rel_rad"]
                                )
                            )
                            for r in rows
                        ),
                        default=0,
                    ),
                    "obstacle_contact_pairs": sorted(collision_pairs),
                    "contact_messages": contact_messages,
                    "contact_frames": sorted(contact_frames),
                    "floor_contact_frames": sorted(contact_with_floor_frames),
                    "sample_count": len(rows),
                }
            )
            field.write_svg(
                folder / "trajectory.svg",
                c,
                [route],
                [(r["x_m"], r["y_m"]) for r in rows],
            )
        text = (
            (folder / "gzserver.log").read_text(errors="replace")
            if (folder / "gzserver.log").exists()
            else ""
        )
        summary["lcp_error_count"] = text.count("LCP internal error")
        if summary["lcp_error_count"]:
            summary["status"] = "invalid_physics"
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 130 if interrupted else (0 if summary["status"] == "goal_reached" else 1)
