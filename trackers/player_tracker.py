# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:30:44 2024

@author: pc
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import csv
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mediapipe as mp
from collections import namedtuple
import math

from utils import detection_utils, draw_utils


#New types to facilitate working with coordinates
Point2D = namedtuple("Point2D", ["x", "y"])
Point3D = namedtuple("Point3D", ["x", "y", "z"])
BBox = namedtuple("BBox", ["x", "y", "w", "h", "conf"])

class PlayerTracker:
    """
    Handles player (skater) pose detection, analysis and visualization using a deep learning model.

    Attributes:
        model_path (str): Local path to the downloaded AI model.
        model : Loaded TensorFlow SavedModel used for pose detection.
        skeleton (str): Skeleton definition used by the model.
        joint_names (np.ndarray): Names of joints for the selected skeleton.
        joint_edges (np.ndarray): Connections between joints (bones).
        frame_count (Integer): Frame counter for identification of each frame.
        segmenter : MediaPipe selfie segmentation model.
        REGION_COLORS (dict): Dictionary for differentiating each region's color on the skeleton.
        JOINT_REGIONS (dict): Dictionary that associates each joint to their respective region.
        GROUND_JOINTS (dict): Dictionary that facilitates finding what joints can be glued to the ground.
    """

    JOINT_REGIONS = {
        "lsho_smpl": "left_arm", "lelb_smpl": "left_arm", "lwri_smpl": "left_arm", "lhan_smpl": "left_arm",
        "rsho_smpl": "right_arm", "relb_smpl": "right_arm", "rwri_smpl": "right_arm", "rhan_smpl": "right_arm",
        "lhip_smpl": "left_leg", "lkne_smpl": "left_leg", "lank_smpl": "left_leg", "ltoe_smpl": "left_leg",
        "rhip_smpl": "right_leg", "rkne_smpl": "right_leg", "rank_smpl": "right_leg", "rtoe_smpl": "right_leg",
        "pelv_smpl": "torso", "bell_smpl": "torso", "spin_smpl": "torso", "thor_smpl": "torso", "neck_smpl": "torso",
        "head_smpl": "head", "htop_mpi_inf_3dhp": "head", "learcoco": "head",
    }

    REGION_COLORS = {
        "left_arm": "#1f77b4",  # blue
        "right_arm": "#ff7f0e",  # orange
        "left_leg": "#2ca02c",  # green
        "right_leg": "#9467bd",  # purple
        "torso": "#8c564b",  # brown
        "head": "#d62728"  # dark red
    }

    GROUND_JOINTS = {"lank_smpl", "rank_smpl", "ltoe_smpl", "rtoe_smpl"}

    def __init__(self,model_type='metrabs_mob3l_y4t', skeleton = 'smpl+head_30'):
        """
        Initializes the PlayerTracker by downloading and loading the pose model,
        configuring the skeleton, and preparing visualization and segmentation tools.

        Args:
            model_type (str): Identifer of the METRABS model to download, uses metrabs mob3l y4t as default
            skeleton (str): Skeleton layout for pose detection, uses smpl+head_30 as default
        """
        self.model_path = detection_utils.download_model(model_type)
        self.model = tf.saved_model.load(self.model_path)
        self.skeleton = skeleton
        self.joint_names = self.model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        self.joint_edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
        self.frame_count = 0

        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects the skater's poses using the AI model on each frame and returns a list of the results.

        Args:
            frames (list[np.ndarray]):  List of frames to process.
            read_from_stub (bool): If true, obtain already detected frames from stub_path.
            stub_path (str): File path used to read/write cached detections.
        Returns:
            list[dict]: The skater's skeleton detections for each frame.
        """
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        self.frame_count = 0

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self,frame):
        """
        Detects the skater's 2D and 3D skeleton on each frame using the downloaded AI model.

        Args:
            frame (np.ndarray): Image frame in RGB format.
        Returns:
            dict: Dictionary indexed by frame number and track ID containing:
              - bounding boxes
              - 2D joint positions
              - 3D joint positions
        """
        input_image = tf.convert_to_tensor(frame, dtype=tf.uint8)

        results = self.model.detect_poses(input_image, skeleton=self.skeleton)

        player_dict = {}
        self.frame_count +=1
        player_dict[self.frame_count] = {}
        track_id = 0
        for boxes, poses2d, poses3d in zip(results['boxes'], results['poses2d'], results['poses3d']):
            track_id += 1
            player_dict[self.frame_count][track_id] = {
                'boxes': boxes.numpy(),
                'poses2d': poses2d.numpy(),
                'poses3d': poses3d.numpy()
            }

        return player_dict

    def draw_results(self, video_frames, player_detections):
        """
        MAIN METHOD OF THE PLAYERTRACKER CLASS:

        Visualizes pose detections, analyzes joint validity, and generates debugging outputs.

        For each frame, this method:
        - Draws bounding boxes and skeleton overlays
        - Validates joint positions using segmentation masks
        - Highlights detection errors
        - Plots the temporal evolution of joint vertical positions
        - Saves annotated frames, logs and CSV reports for offline analysis

        Args:
            video_frames (list[np.ndarray]): Original list of frames obtained from the video.
            player_detections (list[dict]): List of the skater's detections.

        Returns:
            list[np.ndarray]: List of result frames.
        """
        output_video_frames = []
        errors_out_of_bbox = []
        joints_study_data = []

        self._tech_state = {
            "in_jump": False, "jump_frames": [],
            "ground_history": [], "jump_count": 0,
            "total_rotations": 0.0, "jumps": [],
            "airborne_streak": 0, "grounded_streak": 0,
            "all_leg_angles": [],
            "all_body_leans": [],
            "max_leg_angle": 0.0,
            "max_jump_rotations": 0.0,
            "last_yaw": None,
        }

        self.frame_count = 0

        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)

        joints = self.joint_names if hasattr(self, "joint_names") else [str(i) for i in range(24)]

        # Joint's out of the skater's body errors are saved in a log file
        log_file = "joint_errors_log.txt"

        with open(log_file, "w", encoding="utf-8") as log_f:
            log_f.write("Frame,Joint,X,Y,Status\n")

            self._init_figure()

            for frame, player_dict in zip(video_frames, player_detections):
                self.frame_count += 1

                frame_dict = player_dict[self.frame_count]

                for track_id, values in frame_dict.items():


                    #Processing of the joints coordinates into NamedTuples for easier values handling
                    bbox, pose2d, pose3d = detection_utils.process_coordinates(values)

                    #Validates the joints positions, selects their colors and saves their normalized values
                    joint_status, study_entries, bbox_errors = self._validate_joints(frame, pose2d, bbox, joints, track_id)

                    joints_study_data.extend(study_entries)
                    errors_out_of_bbox.extend(bbox_errors)

                    tech = detection_utils.analyze_frame_techniques(pose2d, pose3d, self.frame_count, self.joint_names, self._tech_state)
                    draw_utils.draw_technique_overlay(frame, tech, pose2d, self.joint_names)

                    #Draws each frame with their respective 2D and 3D skeletons, colors and joint evolution graph
                    mat_frame = self._render_full_visualization(frame, bbox, pose3d, joint_status, joints_study_data, output_dir)

                    output_video_frames.append(mat_frame)


        # Log for saving the every point incorrectly place out of the bounding box
        if errors_out_of_bbox:
            csv_path = "errors_out_of_bbox.csv"

            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["frame", "joint", "x", "y", "bbox"]
                )
                writer.writeheader()
                writer.writerows(errors_out_of_bbox)

            print(f"📄 CSV guardado con {len(errors_out_of_bbox)} errores fuera de la bbox")

        print(f"Frames guardados en: {output_dir}")
        print(f"Log articulaciones: {log_file}")

        if self._tech_state["in_jump"] and len(self._tech_state["jump_frames"]) >= 4:
            detection_utils.finalize_jump(self._tech_state)
        self.routine_stats = {
            "jump_count": self._tech_state["jump_count"],
            "total_rotations": round(self._tech_state["total_rotations"], 2),
            "jumps": self._tech_state["jumps"],
            "all_leg_angles": self._tech_state["all_leg_angles"],
            "all_body_leans": self._tech_state["all_body_leans"],
            "max_leg_angle": self._tech_state["max_leg_angle"],
            "max_jump_rotations": self._tech_state["max_jump_rotations"],
        }
        summary_frames = detection_utils.generate_summary_frames(self.routine_stats, output_video_frames[-1].shape)
        output_video_frames.extend(summary_frames)

        return output_video_frames

    def _init_figure(self):
        """
        HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

        Initializes the Matplotlib figure and all visual components used for
        rendering the tracking results.

        This method sets up a multi-panel layout that includes:
        - A main axis for displaying the video frame with 2D pose overlays
        - A 3D axis for visualizing the reconstructed skeleton
        - A graph axis for plotting temporal joint-related metrics

        Additionally, it prepares all reusable artists and placeholders required
        for efficient frame-by-frame updates, including:
        - Image buffer for video frames
        - 2D and 3D skeleton line objects
        - Joint scatter plots (2D and 3D)
        - Bounding box visualization

        The figure is structured using a GridSpec layout to organize spatial
        distribution between image, pose, and analytical plots.

        Returns:
            None
        """
        self.fig = plt.figure(figsize=(14, 7))

        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[2.3, 1],
            height_ratios=[1, 1]
        )

        self.image_ax = self.fig.add_subplot(gs[:, 0])
        self.pose_ax = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.graph_ax = self.fig.add_subplot(gs[1, 1])

        # Placeholders for image
        self.image_artist = self.image_ax.imshow(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        # 2D bones
        self.bone_lines_2d = [
            self.image_ax.plot([], [], linewidth=1.8)[0]
            for _ in self.joint_edges
        ]

        # 3D bones
        self.bone_lines_3d = [
            self.pose_ax.plot([], [], [], )[0]
            for _ in self.joint_edges
        ]

        # 2D joints scatter
        self.joint_scatter_2d = self.image_ax.scatter([], [], s=22)

        # 3D joints scatter
        self.joint_scatter_3d = self.pose_ax.scatter([], [], [], s=5)

        # Bounding box patch
        self.bbox_patch = Rectangle(
            (0, 0), 0, 0,
            fill=False,
            color="yellow",
            linewidth=2,
        )
        self.image_ax.add_patch(self.bbox_patch)

    def _validate_joints(self, frame, pose2d, bbox, joints, track_id):
        """
        HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

        Validates joint positions using segmentation mask and bounding box, assigns their colors and saves their status

        Args:
            frame (np.ndarray): Original frame in BGR format.
            pose2d (list[Point2D]): List of 2D joint coordinates for the current track.
            bbox (BBox or None): Bounding box associated with the detected player.
            joints (list[str] or np.ndarray): List of joint names corresponding to pose indices.
            track_id (int): Identifier of the tracked subject.

        Returns:
            joint_status (dict):
                Dictionary indexed by joint index containing:
                    - 'pt': Point2D coordinate
                    - 'name': Joint name
                    - 'color': Visualization color
                    - 'status': Validation status (OK, OUTSIDE_BODY, OUT_OF_BOUNDS)
            joints_study_entries (list[dict]):
                List of dictionaries containing temporal study information:
                    - 'frame': Frame index
                    - 'joint': Joint index
                    - 'y_norm': Normalized Y coordinate
                    - 'track_id': Track identifier

            errors_out_of_bbox (list[dict]): List of joints detected outside the bounding box, prepared for CSV export.
        """

        joint_status = {}
        joints_study_entries = []
        errors_out_of_bbox = []

        # Convert frame to RGB and generate segmentation mask
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb_frame)
        mask = results.segmentation_mask > 0.3

        # Compute ground reference for normalization
        ground_ys = []
        for joint_idx, pt in enumerate(pose2d):
            joint_name = joints[joint_idx] if joint_idx < len(joints) else None
            if joint_name in self.GROUND_JOINTS:
                ground_ys.append(pt.y)

        ground_y = max(ground_ys) if ground_ys else None

        # Validate each joint
        for joint_idx, pt in enumerate(pose2d):

            joint_name = (
                joints[joint_idx]
                if joint_idx < len(joints)
                else f"joint_{joint_idx}"
            )

            region = self.JOINT_REGIONS.get(joint_name, "torso")
            base_color = self.REGION_COLORS[region]

            # Check if joint is within bbox
            if bbox is not None:
                if not (
                        bbox.x <= pt.x <= bbox.x + bbox.w
                        and bbox.y <= pt.y <= bbox.y + bbox.h
                ):
                    errors_out_of_bbox.append(
                        {
                            "frame": self.frame_count,
                            "joint": joint_name,
                            "x": pt.x,
                            "y": pt.y,
                            "bbox": f"({bbox.x:.1f},{bbox.y:.1f})-({bbox.x + bbox.w:.1f},{bbox.y + bbox.h:.1f})",
                        }
                    )

            # Check if joint is inside body mask
            if 0 <= pt.y < mask.shape[0] and 0 <= pt.x < mask.shape[1]:
                inside = detection_utils.is_inside_body(
                    pt.x, pt.y, mask, kernel=13, thresh=0.10
                )
                color = base_color if inside else "red"
                status = "OK" if inside else "OUTSIDE_BODY"
            else:
                color = "gray"
                status = "OUT_OF_BOUNDS"

            joint_status[joint_idx] = {
                "pt": pt,
                "name": joint_name,
                "color": color,
                "status": status,
            }

            # Normalize Y relative to ground
            if ground_y is not None:
                y_norm = ground_y - pt.y
            else:
                y_norm = 0

            joints_study_entries.append(
                {
                    "frame": self.frame_count,
                    "joint": joint_idx,
                    "y_norm": y_norm,
                    "track_id": track_id,
                }
            )

        return joint_status, joints_study_entries, errors_out_of_bbox

    def _render_full_visualization(self, frame, bbox, pose3d, joint_status, joints_study_data, output_dir):
        """
        HELPER METHOD FOR MAIN METHOD "DRAW_RESULTS":

        Renders full visualization layout for a single frame: 2D skeleton, 3D pose and joint evolution graph.

        Args:
            frame (np.ndarray): BGR image frame.
            bbox (BBox or None): Player bounding box.
            pose3d (list[Point3D] or None): 3D joint coordinates.
            joint_status (dict): Joint visualization info.
            joints_study_data (list[dict]): Temporal joint data.
            output_dir (str): Directory to save rendered frame.

        Returns:
            mat_frame (np.ndarray): Rendered BGR image.
        """

        image_ax = self.image_ax
        pose_ax = self.pose_ax
        graph_ax = self.graph_ax

        # Update main image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image_artist.set_data(image)

        h, w = image.shape[:2]

        self.image_artist.set_extent([0, w, h, 0])

        image_ax.set_xlim(0, w)
        image_ax.set_ylim(h, 0)
        image_ax.tick_params(axis="both", labelsize=8)

        # Draw bounding box
        if bbox is not None:
            self.bbox_patch.set_xy((bbox.x, bbox.y))
            self.bbox_patch.set_width(bbox.w)
            self.bbox_patch.set_height(bbox.h)

        # Setup 3D panel
        if pose3d:
            pose_ax.view_init(5, -75)
            pose_ax.set_xlim3d(-1500, 1500)
            pose_ax.set_zlim3d(-1500, 1500)
            pose_ax.set_ylim3d(2000, 5000)
            pose_ax.set_title("Pose 3D")

        # Draw bones
        for idx, (i_start, i_end) in enumerate(self.joint_edges):

            start = joint_status[i_start]
            end = joint_status[i_end]

            region = self.JOINT_REGIONS.get(start["name"], "torso")
            base_color = self.REGION_COLORS[region]

            color_line = (
                "red"
                if ("red" in (start["color"], end["color"]))
                else base_color
            )

            # 2D line
            line_2d = self.bone_lines_2d[idx]
            line_2d.set_data(
                [start["pt"].x, end["pt"].x],
                [start["pt"].y, end["pt"].y],
            )
            line_2d.set_color(color_line)

            # 3D line
            if pose3d:
                line_3d = self.bone_lines_3d[idx]
                line_3d.set_data(
                    [pose3d[i_start].x, pose3d[i_end].x],
                    [pose3d[i_start].y, pose3d[i_end].y],
                )
                line_3d.set_3d_properties(
                    [pose3d[i_start].z, pose3d[i_end].z]
                )
                line_3d.set_color(color_line)

        # Draw joints scatter
        xs = [j["pt"].x for j in joint_status.values()]
        ys = [j["pt"].y for j in joint_status.values()]
        colors = [j["color"] for j in joint_status.values()]

        self.joint_scatter_2d.set_offsets(np.column_stack([xs, ys]))
        self.joint_scatter_2d.set_color(colors)

        if pose3d:
            self.joint_scatter_3d._offsets3d = (
                [p.x for p in pose3d],
                [p.y for p in pose3d],
                [p.z for p in pose3d],
            )

        # Prepare joint groups for graph
        joint_groups = {}

        for entry in joints_study_data:
            joint_idx = entry["joint"]

            if joint_idx not in joint_groups:
                joint_groups[joint_idx] = {"frame": [], "y_norm": []}

            joint_groups[joint_idx]["frame"].append(entry["frame"])
            joint_groups[joint_idx]["y_norm"].append(entry["y_norm"])

        # Draw graph
        graph_ax.cla()

        for joint_idx, data in joint_groups.items():
            graph_ax.plot(
                data["frame"],
                data["y_norm"],
                alpha=0.6,
                linewidth=1,
            )

        graph_ax.set_title("Evolución Y de articulaciones")
        graph_ax.set_xlabel("Frame")
        graph_ax.set_ylabel("Y")
        graph_ax.axhline(0, color="black", linestyle="--", linewidth=1)

        graph_ax.set_xlim(0, self.frame_count + 1)

        # Render frame
        self.fig.canvas.draw()
        img_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

        # Save frame
        frame_path = os.path.join(
            output_dir, f"frame_{self.frame_count:04d}.jpg"
        )
        cv2.imwrite(frame_path, mat_frame)

        return mat_frame

#Versión original con guardado de frames en carpeta
"""
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        self.frame_count = 0

        output_dir = "frames_debug"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Guardando frames procesados en: {os.path.abspath(output_dir)}")

        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]

            for track_id, values in frame_dict.items():
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)

                bbox = values.get("boxes", None)
                pose2d = values.get("poses2d", None)
                pose3d = values.get("poses3d", None)

                if bbox is not None:
                    x, y, w, h, c = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=1.5))

                if pose3d is not None and pose2d is not None:
                    pose_ax = fig.add_subplot(1, 2, 2, projection="3d")
                    pose_ax.view_init(5, -75)
                    pose_ax.set_xlim3d(-1500, 1500)
                    pose_ax.set_zlim3d(-1500, 1500)
                    pose_ax.set_ylim3d(2000, 5000)

                    pose3d = pose3d.numpy()
                    pose2d = pose2d.numpy()

                    pose3d[..., 1], pose3d[..., 2] = pose3d[..., 2], -pose3d[..., 1]

                    for i_start, i_end in self.joint_edges:
                        image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker="o", markersize=2)
                        pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker="o", markersize=2)

                    image_ax.scatter(*pose2d.T, s=2)
                    pose_ax.scatter(*pose3d.T, s=2)

                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

                output_video_frames.append(mat_frame)

                frame_path = os.path.join(output_dir, f"frame_{self.frame_count:04d}.jpg")
                cv2.imwrite(frame_path, mat_frame)

                plt.close(fig)

        print(f"Se han guardado {self.frame_count} frames en '{output_dir}'")

        return output_video_frames

"""

