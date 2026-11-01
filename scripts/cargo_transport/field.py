#!/usr/bin/env python3
"""Known-map planning and field generation, using only the Python standard library."""
import bisect
import heapq
from html import escape
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .paths import ROOT


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    x0, y0, x1, y1 = c["bounds_m"]
    if not (x0 < x1 and y0 < y1):
        raise ValueError("Invalid field bounds")
    for key in [
        "grid_resolution_m",
        "robot_radius_m",
        "safety_margin_m",
        "path_spacing_m",
    ]:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    ids = set()
    for o in c["obstacles"]:
        a, b, d, e = o["bounds_m"]
        if not (x0 <= a < d <= x1 and y0 <= b < e <= y1) or o["id"] in ids:
            raise ValueError("Obstacle bounds or IDs are invalid")
        ids.add(o["id"])
    for s in c["stations"].values():
        if not point_free(
            s["position_m"], c, c["robot_radius_m"] + c["safety_margin_m"]
        ):
            raise ValueError("Station does not have enough clearance")
    return c


def distance_to_rect(p, rect):
    x, y = p
    x0, y0, x1, y1 = rect
    return math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))


def clearance(p, c, radius=0):
    x, y = p
    x0, y0, x1, y1 = c["bounds_m"]
    values = [x - x0, x1 - x, y - y0, y1 - y]
    values.extend(distance_to_rect(p, o["bounds_m"]) for o in c["obstacles"])
    return min(values) - radius


def point_free(p, c, radius):
    return clearance(p, c, radius) >= -1e-9


def segment_free(a, b, c, radius):
    n = max(1, math.ceil(math.dist(a, b) / (c["grid_resolution_m"] / 4)))
    return all(
        point_free(
            (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n), c, radius
        )
        for i in range(n + 1)
    )


def astar(start, goal, c, radius):
    resolution = c["grid_resolution_m"]
    x0, y0, x1, y1 = c["bounds_m"]

    def cell(p):
        return round((p[0] - x0) / resolution), round((p[1] - y0) / resolution)

    def position(node):
        return x0 + node[0] * resolution, y0 + node[1] * resolution

    s, g = cell(start), cell(goal)
    if not point_free(start, c, radius) or not point_free(goal, c, radius):
        raise ValueError("Start or goal is blocked by the inflated obstacles")
    costs = {s: 0.0}
    parents = {}
    queue = [(math.dist(start, goal), 0.0, s)]
    visited = set()
    while queue:
        _, cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == g:
            route = [node]
            while route[-1] != s:
                route.append(parents[route[-1]])
            points = [position(n) for n in reversed(route)]
            points[0], points[-1] = tuple(start), tuple(goal)
            if not all(
                segment_free(a, b, c, radius) for a, b in zip(points, points[1:])
            ):
                raise ValueError("Snapped endpoints could not be connected safely")
            return points, len(visited)
        p = position(node)
        for dx, dy in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            other = (node[0] + dx, node[1] + dy)
            q = position(other)
            if other in visited or not point_free(q, c, radius):
                continue
            if dx and dy:
                if not point_free(
                    position((node[0] + dx, node[1])), c, radius
                ) or not point_free(position((node[0], node[1] + dy)), c, radius):
                    continue
            if not segment_free(p, q, c, radius):
                continue
            new = cost + math.dist(p, q)
            if new < costs.get(other, float("inf")):
                costs[other] = new
                parents[other] = node
                heapq.heappush(queue, (new + math.dist(q, goal), new, other))
    raise ValueError("No path exists with the configured footprint and clearance")


def simplify(points, c, radius):
    result = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not segment_free(points[i], points[j], c, radius):
            j -= 1
        result.append(points[j])
        i = j
    return result


def round_corners(points, c, radius):
    out = [points[0]]
    for previous, p, next_p in zip(points, points[1:], points[2:]):
        l1, l2 = math.dist(previous, p), math.dist(p, next_p)
        trim = min(0.65, 0.35 * l1, 0.35 * l2)
        accepted = None
        while trim >= 0.015:
            entry = (
                p[0] + (previous[0] - p[0]) * trim / l1,
                p[1] + (previous[1] - p[1]) * trim / l1,
            )
            exit_p = (
                p[0] + (next_p[0] - p[0]) * trim / l2,
                p[1] + (next_p[1] - p[1]) * trim / l2,
            )
            curve = []
            for i in range(31):
                t = i / 30
                curve.append(
                    (
                        (1 - t) ** 2 * entry[0]
                        + 2 * (1 - t) * t * p[0]
                        + t * t * exit_p[0],
                        (1 - t) ** 2 * entry[1]
                        + 2 * (1 - t) * t * p[1]
                        + t * t * exit_p[1],
                    )
                )
            if all(
                segment_free(a, b, c, radius) for a, b in zip([out[-1]] + curve, curve)
            ):
                accepted = curve
                break
            trim /= 2
        if accepted is None:
            raise ValueError(
                "Unable to make a collision-free rounded corner; increase free space"
            )
        out.extend(accepted)
    out.append(points[-1])
    return out


def arc_lengths(points):
    total = [0.0]
    for a, b in zip(points, points[1:]):
        total.append(total[-1] + math.dist(a, b))
    return total


def interpolate(points, lengths, s):
    s = max(0, min(s, lengths[-1]))
    i = min(len(points) - 2, max(0, bisect.bisect_right(lengths, s) - 1))
    distance = lengths[i + 1] - lengths[i]
    ratio = (s - lengths[i]) / distance if distance > 1e-12 else 0
    a, b = points[i], points[i + 1]
    return a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])


def curvature(points, i):
    if not 0 < i < len(points) - 1:
        return 0.0
    a, b, c = points[i - 1 : i + 2]
    den = math.dist(a, b) * math.dist(b, c) * math.dist(a, c)
    return (
        2 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / den
        if den > 1e-10
        else 0.0
    )


def plan(c, start_name, goal_name):
    if start_name == goal_name:
        raise ValueError("Choose different start and goal stations")
    start = c["stations"][start_name]["position_m"]
    goal = c["stations"][goal_name]["position_m"]
    radius = c["robot_radius_m"] + c["safety_margin_m"]
    raw, expanded = astar(start, goal, c, radius + c.get("smoothing_reserve_m", 0))
    simple = simplify(raw, c, radius + c.get("smoothing_reserve_m", 0))
    smooth = round_corners(simple, c, radius)
    lengths = arc_lengths(smooth)
    n = math.ceil(lengths[-1] / c["path_spacing_m"])
    path = [interpolate(smooth, lengths, lengths[-1] * i / n) for i in range(n + 1)]
    if not all(segment_free(a, b, c, radius) for a, b in zip(path, path[1:])):
        raise ValueError("Resampled path violates clearance")
    return {
        "start": start_name,
        "goal": goal_name,
        "points": path,
        "raw_points": raw,
        "simplified_points": simple,
        "length_m": arc_lengths(path)[-1],
        "min_footprint_clearance_m": min(
            clearance(p, c, c["robot_radius_m"]) for p in path
        ),
        "expanded_nodes": expanded,
        "planning_radius_m": radius + c.get("smoothing_reserve_m", 0),
        "max_abs_curvature_per_m": max(
            abs(curvature(path, i)) for i in range(len(path))
        ),
    }


def element(parent, tag, text=None, **attrs):
    result = ET.SubElement(parent, tag, attrs)
    if text is not None:
        result.text = str(text)
    return result


def box_model(world, name, bounds, height, color):
    x0, y0, x1, y1 = bounds
    model = element(world, "model", name=name)
    element(model, "static", "true")
    element(model, "pose", f"{(x0+x1)/2} {(y0+y1)/2} {height/2} 0 0 0")
    link = element(model, "link", name="body")
    for tag in ["collision", "visual"]:
        part = element(link, tag, name=tag)
        box = element(element(part, "geometry"), "box")
        element(box, "size", f"{x1-x0} {y1-y0} {height}")
        if tag == "visual":
            mat = element(part, "material")
            element(mat, "ambient", color)
            element(mat, "diffuse", color)


def disk(link, name, x, y, radius, color, z=0.004):
    visual = element(link, "visual", name=name)
    element(visual, "pose", f"{x} {y} {z} 0 0 0")
    cylinder = element(element(visual, "geometry"), "cylinder")
    element(cylinder, "radius", radius)
    element(cylinder, "length", 0.005)
    mat = element(visual, "material")
    element(mat, "ambient", color)
    element(mat, "diffuse", color)


def write_world(source, target, c, route, contacts=True):
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    floor = world.find("model[@name='floor']/link")
    x0, y0, x1, y1 = c["bounds_m"]
    floor.find("pose").text = f"{(x0+x1)/2} {(y0+y1)/2} -0.05 0 0 0"
    for size in floor.findall(".//geometry/box/size"):
        size.text = f"{x1-x0} {y1-y0} 0.1"
    world.find("gui/camera/pose").text = "6 -4 10 0 0.896 1.57079632679"
    p, q = route["points"][:2]
    yaw = math.atan2(q[1] - p[1], q[0] - p[0])
    robot = world.find("model[@name='cargo_robot']")
    robot.find("pose").text = f"{p[0]} {p[1]} 0.005 0 0 {yaw}"
    cargo = world.find("model[@name='cargo_box']")
    cargo.find("pose").text = f"{p[0]} {p[1]} 0.31 0 0 {yaw}"
    drive = robot.find("plugin[@name='diff_drive']")
    drive.find("max_wheel_acceleration").text = "0"
    drive.find("update_rate").text = "100"
    if contacts:
        for link in robot.findall("link"):
            name = link.get("name")
            sensor = element(link, "sensor", name="contact_" + name, type="contact")
            element(sensor, "always_on", "true")
            element(sensor, "update_rate", 50)
            contact = element(sensor, "contact")
            element(contact, "collision", link.find("collision").get("name"))
            plugin = element(
                sensor,
                "plugin",
                name="contact_plugin_" + name,
                filename="libgazebo_ros_bumper.so",
            )
            ros = element(plugin, "ros")
            element(ros, "namespace", "/cargo_robot")
            element(ros, "remapping", "bumper_states:=contacts")
            element(plugin, "frame_name", name)
    for obstacle in c["obstacles"]:
        box_model(
            world,
            "obstacle_" + obstacle["id"],
            obstacle["bounds_m"],
            obstacle["height_m"],
            "0.35 0.40 0.48 1",
        )
    wall = 0.15
    for name, bounds in [
        ("west", [x0 - wall, y0 - wall, x0, y1 + wall]),
        ("east", [x1, y0 - wall, x1 + wall, y1 + wall]),
        ("south", [x0, y0 - wall, x1, y0]),
        ("north", [x0, y1, x1, y1 + wall]),
    ]:
        box_model(world, "boundary_" + name, bounds, 0.65, "0.7 0.7 0.72 1")
    markers = element(world, "model", name="floor_markings")
    element(markers, "static", "true")
    link = element(markers, "link", name="markings")
    for name, station in c["stations"].items():
        x, y = station["position_m"]
        disk(link, "station_" + name, x, y, 0.42, "0.15 0.55 0.65 1")
    for i, p in enumerate(route["points"][::3]):
        disk(link, f"path_{i}", p[0], p[1], 0.025, "0.1 0.8 0.25 1", 0.009)
    ET.indent(tree, space="  ")
    tree.write(target, encoding="utf-8", xml_declaration=True)


def write_svg(target, c, routes, trajectory=None):
    scale = 65
    margin = 65
    x0, y0, x1, y1 = c["bounds_m"]
    width = (x1 - x0) * scale + 2 * margin
    height = (y1 - y0) * scale + 2 * margin + 45

    def xy(p):
        return margin + (p[0] - x0) * scale, margin + (y1 - p[1]) * scale

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial,sans-serif" font-size="14">',
        f'<text x="{margin}" y="28" font-size="20" font-weight="bold">Factory parts transport / 12 m x 8 m</text>',
    ]
    for i in range(math.floor(x0), math.ceil(x1) + 1):
        a, b = xy((i, y0)), xy((i, y1))
        out.append(f'<path d="M {a[0]} {a[1]} L {b[0]} {b[1]}" stroke="#e4e9ee"/>')
        out.append(f'<text x="{a[0]-5}" y="{a[1]+22}">{i}</text>')
    for i in range(math.floor(y0), math.ceil(y1) + 1):
        a, b = xy((x0, i)), xy((x1, i))
        out.append(f'<path d="M {a[0]} {a[1]} L {b[0]} {b[1]}" stroke="#e4e9ee"/>')
        out.append(f'<text x="{a[0]-22}" y="{a[1]+5}">{i}</text>')
    a = xy((x0, y1))
    out.append(
        f'<rect x="{a[0]}" y="{a[1]}" width="{(x1-x0)*scale}" height="{(y1-y0)*scale}" fill="none" stroke="#283747" stroke-width="3"/>'
    )
    for o in c["obstacles"]:
        a, b, d, e = o["bounds_m"]
        x, y = xy((a, e))
        out.append(
            f'<rect x="{x}" y="{y}" width="{(d-a)*scale}" height="{(e-b)*scale}" rx="3" fill="#526274"/>'
        )
        out.append(
            f'<text x="{x+(d-a)*scale/2}" y="{y+(e-b)*scale/2}" text-anchor="middle" fill="white">{escape(o["label"])}</text>'
        )
    colors = ["#d36a16", "#2066c1", "#8b46a8"]
    for i, route in enumerate(routes):
        points = " ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in route["points"])
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[i%len(colors)]}" stroke-width="3"/>'
        )
        out.append(
            f'<text x="{margin+i*260}" y="{height-20}" fill="{colors[i%len(colors)]}">{route["start"]} to {route["goal"]}: {route["length_m"]:.2f} m</text>'
        )
    if trajectory:
        points = " ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in trajectory)
        out.append(
            f'<polyline points="{points}" fill="none" stroke="#00a486" stroke-width="2" stroke-dasharray="5 3"/>'
        )
    for name, s in c["stations"].items():
        x, y = xy(s["position_m"])
        out.append(f'<circle cx="{x}" cy="{y}" r="13" fill="#083d54"/>')
        out.append(
            f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="white" font-weight="bold">{name}</text>'
        )
        out.append(
            f'<text x="{x}" y="{y+31}" text-anchor="middle">{escape(s["label"])}</text>'
        )
    out.append("</g></svg>")
    Path(target).write_text("\n".join(out), encoding="utf-8")
