# -*- coding: utf-8 -*-
"""
read videos

@author: pc
"""

import cv2
import numpy as np
import math

def draw_technique_overlay(frame, tech, pose2d, joint_names):
    """
    HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

    Draws technique-related textual information on top of the frame.

    This method overlays key biomechanical metrics extracted from the current
    frame analysis, including:
    - Leg angle between lower limbs
    - Body lean angle
    - Jump detection status, including height and accumulated rotations

    The information is rendered as readable text with background support to
    ensure visibility regardless of the underlying image content.

    Additionally, it delegates the rendering of graphical indicators such as
    arrows and angle arcs to the corresponding helper method.

    Args:
        frame (np.ndarray): Current video frame (BGR) to annotate in-place.
        tech (dict): Dictionary containing computed technique metrics for
            the current frame.
        pose2d (list[Point2D]): List of 2D joint coordinates.
        joint_names (list[str]): List of joint names used for indexing.

    Returns:
        None
    """
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1
    WHITE = (255, 255, 255)
    YELLOW = (0, 220, 255)
    RED = (0, 0, 255)
    y = 30 # Initial vertical offset for stacked text lines

    # Displays leg angle if available
    if tech["leg_angle"] is not None:
        draw_text_with_bg(frame, f"Leg angle: {tech['leg_angle']:.1f} deg",
                                (10, y), font, scale, WHITE, thick)
    else:
        draw_text_with_bg(frame, "Leg angle: --",
                          (10, y), font, scale, RED, thick)

    y += 26

    # Displays body lean if available
    if tech["body_lean"] is not None:
        draw_text_with_bg(frame, f"Body lean: {tech['body_lean']:.1f} deg",
                                (10, y), font, scale, WHITE, thick)
    else:
        draw_text_with_bg(frame, "Body lean: --",
                          (10, y), font, scale, RED, thick)

    y += 26

    # Highlights jump state and related metrics
    if tech["is_airborne"]:
        draw_text_with_bg(frame, f"JUMP  H:{tech['height_px']:.0f}px",
                                (10, y), font, scale, YELLOW, 2)
        y += 28
        draw_text_with_bg(frame, f"Rot: {tech['rotations_so_far']:.2f} rev",
                                (10, y), font, scale, YELLOW, thick)

    # Draws graphical indicators (arrow, arcs, etc.)
    draw_technique_indicators(frame, pose2d, tech, joint_names)


def draw_text_with_bg(frame, text, pos, font, scale, text_color, thickness,
                       bg_color=(0, 0, 0), alpha=0.55):
    """
    HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

    Draws text on the frame with a semi-transparent background rectangle
    to improve readability.

    This method computes the text bounding box, renders a padded background
    rectangle using alpha blending, and then overlays the text on top.

    It is used to ensure that textual information remains visible regardless
    of variations in the underlying image.

    Args:
        frame (np.ndarray): Frame where the text will be drawn (modified in-place).
        text (str): Text string to render.
        pos (tuple[int, int]): Bottom-left corner of the text (x, y).
        font (int): OpenCV font type.
        scale (float): Font scale factor.
        text_color (tuple[int, int, int]): Text color in BGR format.
        thickness (int): Thickness of the text strokes.
        bg_color (tuple[int, int, int], optional): Background rectangle color.
        alpha (float, optional): Transparency factor for the background.

    Returns:
        None
    """
    # Computes text size and baseline for proper bounding box placement
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = pos # Bottom-left reference point for text
    overlay = frame.copy()  # Creates overlay for alpha blending

    pad = 4 # Padding around text box

    # Draws background rectangle on overlay
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg_color, -1)

    # Blends overlay with original frame for transparency effect
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Renders final text on top of the background
    cv2.putText(frame, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)


def draw_technique_indicators(frame, pose2d, tech, joint_names):
    """
    HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

    Draws graphical technique indicators directly on the frame to complement
    textual information.

    This method visualizes key biomechanical cues using geometric primitives:
    - Jump indicator: A vertical arrow originating from the pelvis and scaled
      proportionally to the detected jump height
    - Leg angle visualization: An arc representing the angle between both legs,
      computed from hip-to-knee vectors, along with supporting lines

    These indicators provide an intuitive visual interpretation of movement
    dynamics such as airtime and lower-body configuration.

    Internal helper functions are used to safely retrieve joint indices and
    coordinates based on joint names.

    Args:
        frame (np.ndarray): BGR frame to annotate in-place.
        pose2d (list[Point2D]): List of 2D joint coordinates.
        tech (dict): Dictionary containing technique analysis results.
        joint_names (list[str]): List of joint names used for indexing.

    Returns:
        None
    """
    # Retrieves joint index by name safely
    def jidx(name):
        matches = np.where(joint_names == name)[0]
        return int(matches[0]) if len(matches) > 0 else None

    # Retrieves 2D joint safely from pose list
    def get2(name):
        idx = jidx(name)
        if idx is None or pose2d is None or idx >= len(pose2d): return None
        return pose2d[idx]

    YELLOW = (0, 220, 255)
    CYAN = (255, 200, 0)
    WHITE = (255, 255, 255)

    # Jump arrow visualization
    if tech["is_airborne"]:
        pelv = get2("pelv_smpl")
        if pelv is not None:
            height_val = tech["height_px"]
            arrow_len = int(min(height_val * 0.4, 90)) # Scales arrow length
            base = (int(pelv.x), int(pelv.y)) # Arrow base position
            tip_y = max(int(pelv.y - arrow_len), 5)
            tip = (int(pelv.x), tip_y) # Arrow tip position

            if arrow_len > 5:
                # Draws vertical arrow representing jump height
                cv2.arrowedLine(frame, base, tip, YELLOW, 3,
                                cv2.LINE_AA, tipLength=0.25)

                # Displays height value next to arrow
                text_y = max(int(pelv.y - arrow_len // 2), 15)
                cv2.putText(frame, f"{height_val:.0f}px",
                            (int(pelv.x) + 10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

    # Leg angle arc visualization
    if tech["leg_angle"] is not None:
        lhip = get2("lhip_smpl")
        rhip = get2("rhip_smpl")
        lkne = get2("lkne_smpl")
        rkne = get2("rkne_smpl")

        # Ensures all required joints are available
        if None not in (lhip, rhip, lkne, rkne):

            # Computes midpoint between hips as reference origin
            mx = (lhip.x + rhip.x) // 2
            my = (lhip.y + rhip.y) // 2

            # Computes leg vectors from hips to knees
            vl = (lkne.x - lhip.x, lkne.y - lhip.y)
            vr = (rkne.x - rhip.x, rkne.y - rhip.y)

            # Converts vectors to angles in degrees
            angle_l = math.degrees(math.atan2(vl[1], vl[0]))
            angle_r = math.degrees(math.atan2(vr[1], vr[0]))

            # Computes angular difference and always selects the inner arc
            diff = angle_r - angle_l
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            # Start is always angle_l, end is angle_l + diff (inner arc)
            arc_start = angle_l
            arc_end = angle_l + diff

            radius = 38 # Radius of arc visualization

            # Draws arc representing angle between both legs
            cv2.ellipse(frame,
                        (mx, my),
                        (radius, radius),
                        0,
                        arc_start,
                        arc_end,
                        CYAN, 2, cv2.LINE_AA)

            # Draws helper lines from hip midpoint to each knee
            cv2.line(frame, (mx, my), (int(lkne.x), int(lkne.y)), CYAN, 1, cv2.LINE_AA)
            cv2.line(frame, (mx, my), (int(rkne.x), int(rkne.y)), CYAN, 1, cv2.LINE_AA)

            # Computes position for angle label
            mid_angle = math.radians((angle_l + angle_r) / 2)
            lx = int(mx + (radius + 14) * math.cos(mid_angle))
            ly = int(my + (radius + 14) * math.sin(mid_angle))

            # Displays leg angle value near arc
            cv2.putText(frame, f"{tech['leg_angle']:.0f} deg",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, WHITE, 1, cv2.LINE_AA)
