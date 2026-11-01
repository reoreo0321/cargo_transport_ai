"""Cargo inertia/friction settings and state geometry; simulation only."""

import math
import xml.etree.ElementTree as ET


def angles(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return (roll, pitch, yaw)


def slope(points, t, value, window):
    points.append((t, value))
    while len(points) > 2 and t - points[0][0] > window:
        points.popleft()
    if len(points) < 2 or points[-1][0] - points[0][0] < 0.015:
        return None
    mt = sum((p[0] for p in points)) / len(points)
    mv = sum((p[1] for p in points)) / len(points)
    denom = sum(((p[0] - mt) ** 2 for p in points))
    return (
        sum(((p[0] - mt) * (p[1] - mv) for p in points)) / denom
        if denom > 1e-12
        else None
    )


def apply_cargo(world_path, condition):
    """60% uniform cuboid + 40% small ballast sphere, consistent COM/inertia."""
    tree = ET.parse(world_path)
    cargo = tree.getroot().find("world/model[@name='cargo_box']")
    pose = cargo.find("pose").text.split()
    (sx, sy, sz) = condition["size_m"]
    pose[2] = str(0.25 + sz / 2)
    cargo.find("pose").text = " ".join(pose)
    for size in cargo.findall(".//geometry/box/size"):
        size.text = f"{sx} {sy} {sz}"
    link = cargo.find("link")
    mass = condition["mass_kg"]
    center = condition["com_z_fraction"] * sz
    ballast_z = center / 0.4
    inertia = link.find("inertial")
    inertia.find("mass").text = str(mass)
    node = inertia.find("pose")
    if node is None:
        node = ET.SubElement(inertia, "pose")
    node.text = f"0 0 {center} 0 0 0"
    parallel = 0.6 * mass * center**2 + 0.4 * mass * (ballast_z - center) ** 2
    sphere = 0.4 * mass * 2 * 0.015**2 / 5
    diagonal = [
        0.6 * mass * (sy * sy + sz * sz) / 12 + parallel + sphere,
        0.6 * mass * (sx * sx + sz * sz) / 12 + parallel + sphere,
        0.6 * mass * (sx * sx + sy * sy) / 12 + sphere,
    ]
    for key, value in zip(["ixx", "iyy", "izz"], diagonal):
        inertia.find("inertia/" + key).text = str(value)
    for tag in ["mu", "mu2"]:
        link.find("collision/surface/friction/ode/" + tag).text = str(condition["mu"])
    tray = tree.getroot().find(
        "world/model[@name='cargo_robot']/link[@name='base_link']"
    )
    for tag in ["mu", "mu2"]:
        tray.find("collision/surface/friction/ode/" + tag).text = "1.0"
    ET.indent(tree, space="  ")
    tree.write(world_path, encoding="utf-8", xml_declaration=True)
    return {
        "mass_kg": mass,
        "com_z_m": center,
        "inertia_diagonal_kg_m2": diagonal,
        "tray_mu": 1.0,
        "cargo_mu": condition["mu"],
    }
