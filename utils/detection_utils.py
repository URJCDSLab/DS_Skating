# -*- coding: utf-8 -*-
"""
read videos

@author: pc
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import tensorflow as tf
import zipfile
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import math

#New types to facilitate working with coordinates
Point2D = namedtuple("Point2D", ["x", "y"])
Point3D = namedtuple("Point3D", ["x", "y", "z"])
BBox = namedtuple("BBox", ["x", "y", "w", "h", "conf"])

def download_model(model_type):
    """
    Downloads and extracts the AI model for the skeleton's pose detection.

    Args:
        model_type (str): Downloads the specified model type.
    Returns:
        str: The path of the recently saved model.
    """
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'

    # Downloads model ZIP file (cached if already exists)
    model_zip_path = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}_20211019.zip',
        cache_subdir='models',
        extract=False)

    # Defines expected extraction path
    model_extract_path = os.path.join(os.path.dirname(model_zip_path), model_type)

    # Extracts model only if not already available
    if not os.path.exists(model_extract_path):
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(model_zip_path))

        # Renames extracted folder to a cleaner path
        extracted_folder = os.path.join(os.path.dirname(model_zip_path), f"{model_type}_20211019")
        if os.path.exists(extracted_folder):
            os.rename(extracted_folder, model_extract_path)

    return model_extract_path


def is_inside_body(x, y, mask, kernel=19, thresh=0.00):
    """
    Determines if a 2D joint coordinate resides within the subject's segmentation mask.

    Uses a distance-weighted neighborhood analysis to handle boundary uncertainty
    and prevent false negatives near the skater's edges.

    Args:
        x (float or int): Horizontal coordinate of the joint.
        y (float or int): Vertical coordinate of the joint.
        mask (np.ndarray): Binary or probability segmentation mask.
        kernel (int): Window size for local neighborhood analysis. Defaults to 19.
        thresh (float): Confidence threshold for the weighted score. Defaults to 0.00.

    Returns:
        bool: True if the joint is validated inside the body, False otherwise.
    """
    h, w = mask.shape

    # Validates that coordinates fall within the image boundary limits
    if not (0 <= x < w and 0 <= y < h):
        return False

    # Returns true immediately if the exact coordinate hits the mask
    if mask[int(y), int(x)]:
        return True

    # Extracts the local neighborhood bounding box around the coordinate
    y1, y2 = max(0, int(y - kernel)), min(h, int(y + kernel + 1))
    x1, x2 = max(0, int(x - kernel)), min(w, int(x + kernel + 1))
    area = mask[y1:y2, x1:x2]

    # Prevents processing if the cropped neighborhood area is empty
    if area.size == 0:
        return False

    # Generates a coordinate grid to compute spatial distances
    y_indices, x_indices = np.mgrid[y1:y2, x1:x2]
    dist_al_centro = np.sqrt((y_indices - y) ** 2 + (x_indices - x) ** 2)

    # Computes exponential decay weights based on distance to the center
    pesos = np.exp(-dist_al_centro / (kernel / 2))

    # Normalizes the weight matrix to sum up to 1.0
    pesos_norm = pesos / np.sum(pesos)

    # Evaluates the final weighted score within the neighborhood
    score_ponderado = np.sum(area * pesos_norm)

    return score_ponderado > thresh


def process_coordinates(values):
    """
    Converts raw numpy pose outputs into structured custom coordinate types: Point2D, Point3D and BBox.

    Args:
        values (dict): Dictionary containing raw 'boxes', 'poses2d' and 'poses3d'.

    Returns:
        tuple:
        bbox (BBox or None): Structured bounding box object if available.
        pose2d (list[Point2D] or None): List of structured 2D joint coordinates.
        pose3d (list[Point3D] or None): List of structured 3D joint coordinates with axis adjustment.
    """
    raw_bbox = values.get("boxes", None)
    raw_pose2d = values.get("poses2d", None)
    raw_pose3d = values.get("poses3d", None)

    bbox = None
    pose2d = None
    pose3d = None

    # Process bounding box
    if raw_bbox is not None:
        bbox = BBox(*raw_bbox)

    # Process 2D joints
    if raw_pose2d is not None:
        pose2d = [Point2D(int(x), int(y)) for x, y in raw_pose2d.numpy()]

    # Process 3D joints
    if raw_pose3d is not None:
        p3d = raw_pose3d.numpy()

        # Adjust Y-Z axes for visualization
        p3d[..., 1], p3d[..., 2] = p3d[..., 2], -p3d[..., 1]
        pose3d = [Point3D(x, y, z) for x, y, z in p3d]

    return bbox, pose2d, pose3d

def analyze_frame_techniques(pose2d, pose3d, frame_idx, joint_names, tech_state, is_valid=True):
    """
    HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

    Analyzes skating techniques for a single frame and updates internal jump state.

    Must be called once per frame in order. Relies on self._tech_state (initialized
    by draw_results before the loop) to persist jump tracking across frames.

    Args:
        pose2d (list[Point2D]): 2D joint coordinates for this frame.
        pose3d (list[Point3D]): 3D joint coordinates for this frame.
        frame_idx (int): Current frame index (used for jump bookkeeping).

    Returns:
        dict: Per-frame metrics ready for overlay:
            - leg_angle (float|None)
            - body_lean (float|None)
            - is_airborne (bool)
            - height_px (float)
            - rotations_so_far (float)  # rotations accumulated in current jump
    """

    # Retrieves joint index by name
    def jidx(name):
        matches = np.where(joint_names == name)[0]
        return int(matches[0]) if len(matches) > 0 else None

    # Retrieves 2D joint safely
    def get2(name):
        idx = jidx(name)
        if idx is None or pose2d is None or idx >= len(pose2d): return None
        return pose2d[idx]

    # Retrieves 3D joint safely
    def get3(name):
        idx = jidx(name)
        if idx is None or pose3d is None or idx >= len(pose3d): return None
        return pose3d[idx]

    # Computes angle between two 2D vectors
    def angle_between(v1, v2):
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

    if not is_valid:
        tech_state["last_yaw"] = None

        return {
            "leg_angle": None,
            "body_lean": None,
            "is_airborne": False,
            "height_px": 0.0,
            "rotations_so_far": 0.0,
        }


    # Leg angle (hip→knee vectors)
    lhip, rhip = get2("lhip_smpl"), get2("rhip_smpl")
    lkne, rkne = get2("lkne_smpl"), get2("rkne_smpl")
    leg_angle = None
    if None not in (lhip, rhip, lkne, rkne):
        vl = (lkne.x - lhip.x, lkne.y - lhip.y)
        vr = (rkne.x - rhip.x, rkne.y - rhip.y)
        raw_angle = angle_between(vl, vr)

        # Cross product z-component: positive = left knee is to the left of right knee
        # If negative, vectors are crossed and we are measuring the exterior angle
        cross = vl[0] * vr[1] - vl[1] * vr[0]
        if cross < 0:
            raw_angle = 360 - raw_angle

        # Inner angle between legs must always be in [0, 180]
        leg_angle = min(raw_angle, 360 - raw_angle)

    # Body lean (pelvis→thorax vs vertical)
    pelv, thor = get2("pelv_smpl"), get2("thor_smpl")
    body_lean = None
    if None not in (pelv, thor):
        body_lean = angle_between((thor.x - pelv.x, thor.y - pelv.y), (0, -1))

    # Jump height estimation (pelvis vs ankles)
    pelv_pt = get2("pelv_smpl")

    ankle_pts = [get2(k) for k in ("lank_smpl", "rank_smpl")]
    ankle_pts = [p for p in ankle_pts if p is not None]

    if ankle_pts and pelv_pt:
        ankle_y = sum(p.y for p in ankle_pts) / len(ankle_pts)
        height_px = ankle_y - pelv_pt.y # Positive = pelvis above ankles
    else:
        height_px = 0.0

    # Determines airborne state using height threshold
    is_airborne = height_px > 180

    # Rotation estimation using shoulder yaw (3D)
    ls, rs = get3("lsho_smpl"), get3("rsho_smpl")
    yaw = math.degrees(math.atan2(rs.z - ls.z, rs.x - ls.x)) \
        if None not in (ls, rs) else None

    st = tech_state

    if yaw is not None:
        if st["last_yaw"] is not None:
            d = yaw - st["last_yaw"]
            if d > 180:
                d -= 360
            elif d < -180:
                d += 360
            st["total_rotations"] += abs(d) / 360.0
        st["last_yaw"] = yaw

    # Jump state machine (temporal filtering)
    JUMP_ENTER_FRAMES = 3
    JUMP_EXIT_FRAMES = 3

    # Updates streak counters for airborne / grounded states
    if is_airborne:
        st["airborne_streak"] += 1
        st["grounded_streak"] = 0
    else:
        st["grounded_streak"] += 1
        st["airborne_streak"] = 0

    # Applies temporal smoothing to avoid noisy detections
    confirmed_airborne = st["airborne_streak"] >= JUMP_ENTER_FRAMES
    confirmed_grounded = st["grounded_streak"] >= JUMP_EXIT_FRAMES

    # Handles jump start and accumulation
    if confirmed_airborne:
        if not st["in_jump"]:
            st["in_jump"] = True
            st["jump_frames"] = []
        st["jump_frames"].append({
            "frame": frame_idx, "height": height_px,
            "lean": body_lean, "yaw": yaw, "leg_ang": leg_angle
        })

    # Handles jump end and finalization
    elif confirmed_grounded and st["in_jump"]:
        if len(st["jump_frames"]) >= 4:
            finalize_jump(tech_state)
        st["in_jump"] = False
        st["jump_frames"] = []

    # Handles transition frames (still part of jump)
    elif st["in_jump"]:
        # Todavía en transición, sigue acumulando
        st["jump_frames"].append({
            "frame": frame_idx, "height": height_px,
            "lean": body_lean, "yaw": yaw, "leg_ang": leg_angle
        })

    # Live rotation tracking during jump
    yaws_so_far = [f["yaw"] for f in st["jump_frames"] if f["yaw"] is not None]
    rot_so_far = 0.0
    if len(yaws_so_far) > 1:
        total_delta = 0.0
        for i in range(1, len(yaws_so_far)):
            d = yaws_so_far[i] - yaws_so_far[i - 1]
            if d > 180:
                d -= 360
            elif d < -180:
                d += 360
            total_delta += d
        rot_so_far = abs(total_delta) / 360.0

    if leg_angle is not None:
        st["all_leg_angles"].append(leg_angle)
        if leg_angle > st["max_leg_angle"]:
            st["max_leg_angle"] = leg_angle
    if body_lean is not None:
        st["all_body_leans"].append(body_lean)

    return {
        "leg_angle": leg_angle,
        "body_lean": body_lean,
        "is_airborne": is_airborne,
        "height_px": height_px,
        "rotations_so_far": rot_so_far,
    }

def finalize_jump(tech_state):
    """
    HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

    Finalizes the current jump sequence by analyzing the accumulated airborne
    frames and storing a summarized entry in the jump log.

    This method:
    - Computes total body rotation during the jump using yaw angle differences
      (handling angular wrap-around at ±180 degrees)
    - Estimates the number of rotations performed in the air
    - Extracts key jump metrics such as:
        * Start and end frames
        * Maximum height reached
        * Mean body lean angle
        * Mean leg separation angle
    - Updates global technique state statistics, including total jump count
      and accumulated rotations

    The resulting jump summary is appended to the "jumps" list inside the
    technique state dictionary.

    Args:
        tech_state (dict): Dictionary containing the temporal state of the
            technique analysis, including accumulated frame data for the
            current jump.

    Returns:
        None
    """

    st = tech_state # Alias for readability
    jf = st["jump_frames"]  # List of frames belonging to the current jump

    # Extracts valid yaw values across the jump
    yaws = [f["yaw"] for f in jf if f["yaw"] is not None]

    total_delta = 0.0  #Initial degrees of the jump

    # Computes yaw differences frame-to-frame with wrap-around correction
    for i in range(1, len(yaws)):
        d = yaws[i] - yaws[i - 1]

        # Corrects discontinuities at ±180 degrees
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360

        total_delta += d # Accumulates rotation

    # Degrees to nRotations
    rotations = abs(total_delta) / 360.0

    if rotations > st["max_jump_rotations"]:
        st["max_jump_rotations"] = rotations

    st["jump_count"] += 1
    st["jumps"].append({
        "jump_id": st["jump_count"],
        "start_frame": jf[0]["frame"],
        "end_frame": jf[-1]["frame"],
        "max_height_px": max(f["height"] for f in jf),
        "rotations": round(rotations, 2),

        # Computes mean body lean ignoring missing values
        "mean_lean_deg": float(np.mean([f["lean"] for f in jf if f["lean"] is not None] or [0])),

        # Computes mean leg angle ignoring missing values
        "mean_leg_angle_deg": float(np.mean([f["leg_ang"] for f in jf if f["leg_ang"] is not None] or [0])),
    })

def generate_summary_frames(stats, frame_shape, n_frames=120):
    """
    Generates a sequence of static summary frames to append at the end of the video.

    Args:
        stats (dict): Output from self.routine_stats (populated by draw_results).
        frame_shape (tuple): (height, width, channels) of the video frames.
        n_frames (int): Number of frames to hold the summary screen (default ~3s at 30fps).

    Returns:
        list[np.ndarray]: List of BGR summary frames.
    """

    # Extracts aggregated statistics
    jumps = stats.get("jumps", [])
    jump_count = stats.get("jump_count", 0)
    total_rotations = stats.get("total_rotations", 0.0)

    # Computes averages safely
    avg_rot = (total_rotations / jump_count) if jump_count > 0 else 0.0
    avg_lean = float(np.mean([j["mean_lean_deg"] for j in jumps]) if jumps
                     else np.mean(stats.get("all_body_leans", [0]) or [0]))
    avg_h = float(np.mean([j["max_height_px"] for j in jumps]) if jumps else 0)
    avg_leg = float(np.mean([j["mean_leg_angle_deg"] for j in jumps]) if jumps
                    else np.mean(stats.get("all_leg_angles", [0]) or [0]))
    max_leg = stats.get("max_leg_angle", 0.0)
    max_jump_rot = stats.get("max_jump_rotations", 0.0)

    # Score computation based on performance metrics
    score = 0
    score += min(jump_count * 5, 30)
    score += min(avg_rot * 10, 20)
    score += min(max_jump_rot * 5, 10)
    score += 15 if avg_lean <= 15 else (10 if avg_lean <= 40 else 5 if avg_lean <= 60 else 2)
    score += 10 if avg_h >= 80 else (6 if avg_h >= 40 else 0)
    score += 20 if avg_leg >= 80 else (14 if avg_leg >= 50 else 7 if avg_leg >= 10 else 0)
    score += 20 if max_leg >= 150 else (14 if max_leg >= 90 else 7 if max_leg >= 45 else 0)
    score += min(int(total_rotations) * 2, 20)
    score = int(round(min(score, 100)))

    # Assigns grade and color based on score
    if score >= 85:
        grade, grade_color = "A", (0.0, 0.82, 0.0)
    elif score >= 70:
        grade, grade_color = "B", (0.0, 0.78, 1.0)
    elif score >= 55:
        grade, grade_color = "C", (0.78, 0.78, 0.0)
    elif score >= 40:
        grade, grade_color = "D", (1.0, 0.55, 0.0)
    else:
        grade, grade_color = "F", (0.86, 0.0, 0.0)

    # Figure and layout setup
    h, w = frame_shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0d0d1a")

    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], left=0.06,
                          right=0.97, top=0.92, bottom=0.10, wspace=0.35)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # Applies consistent dark theme
    for ax in (ax_left, ax_right):
        ax.set_facecolor("#0d0d1a")

    ax_left.axis("off")

    #  Summary text panel
    ax_left.text(0.0, 1.02, "ROUTINE SUMMARY",
                 transform=ax_left.transAxes,
                 fontsize=15, fontweight="bold", color="white", va="bottom")

    # Key metrics rows
    rows = [
        ("Jumps detected", f"{jump_count}"),
        ("Total rotations", f"{total_rotations:.2f} rev"),
        ("Avg rotations/jump", f"{avg_rot:.2f} rev"),
        ("Avg jump height", f"{avg_h:.0f} px"),
        ("Avg body lean", f"{avg_lean:.1f}°"),
        ("Avg leg angle", f"{avg_leg:.1f}°"),
        ("Max leg angle", f"{max_leg:.1f}°"),
        ("Max rotations/jump", f"{max_jump_rot:.2f} rev"),
    ]

    # Renders rows with consistent spacing
    row_h = 0.11
    y0 = 0.92
    for i, (label, value) in enumerate(rows):
        y = y0 - i * row_h
        ax_left.text(0.02, y, f"{label}:", transform=ax_left.transAxes,
                     fontsize=11, color="white", fontweight="bold", va="center")
        ax_left.text(0.62, y, value, transform=ax_left.transAxes,
                     fontsize=11, color="#a0d4ff", va="center")

    # Divider line
    ax_left.plot([0.0, 1.0], [0.08, 0.08], color="#333355", linewidth=1,
                 transform=ax_left.transAxes)

    # Final score and grade display
    ax_left.text(0.02, 0.05, f"SCORE:  {score} / 100  →  {grade}",
                 transform=ax_left.transAxes,
                 fontsize=16, fontweight="bold", color=grade_color, va="center")

    # Right panel: per-jump bar chart
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_color("#444")
    ax_right.spines["left"].set_color("#444")
    ax_right.tick_params(colors="white")
    ax_right.yaxis.label.set_color("white")
    ax_right.xaxis.label.set_color("white")
    ax_right.set_title("Per-jump breakdown", color="white", fontsize=11, pad=8)

    if jumps:

        # Extracts per-jump metrics
        ids = [f"J{j['jump_id']}" for j in jumps]
        heights = [j["max_height_px"] for j in jumps]
        rots = [j["rotations"] for j in jumps]
        x = np.arange(len(ids))

        # Draws bar chart (jump height)
        bars = ax_right.bar(x, heights, color="#3a7fd5", alpha=0.85, zorder=3)
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(ids, color="white", fontsize=10)
        ax_right.set_ylabel("Max height (px)", fontsize=10)

        if heights:
            ax_right.set_ylim(0, max(heights) * 1.2)

        # Annotates each bar with rotation count
        for bar, rot in zip(bars, rots):
            current_height = bar.get_height()
            margin = max(current_height * 0.04, 2)

            ax_right.text(
                bar.get_x() + bar.get_width() / 2,
                current_height + margin,
                f"{rot:.1f} rev",
                ha="center", va="bottom",
                color="#f0c060", fontsize=9, fontweight="bold"
            )
    else:

        # Displays fallback message if no jumps detected
        ax_right.text(0.5, 0.5, "No jumps detected",
                      ha="center", va="center", color="gray",
                      fontsize=12, transform=ax_right.transAxes)
        ax_right.axis("off")

    # Convert figure to OpenCV frame
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    panel = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    panel = cv2.resize(panel, (w, h))
    plt.close(fig)

    return [panel] * n_frames
