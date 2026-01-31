# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:30:44 2024

@author: pc
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "2"

import zipfile
import cv2
import csv
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mediapipe as mp
import pandas as pd
from collections import namedtuple

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
        region_colors (dict): Dictionary for differentiating each region's color on the skeleton.
        region_map (dict): Dictionary that associates each joint to their respective region.
        ground_joints (dict): Dictionary that facilitates finding what joints can be glued to the ground.
    """

    def __init__(self,model_type='metrabs_mob3l_y4t', skeleton = 'smpl+head_30'):
        """
        Initializes the PlayerTracker by downloading and loading the pose model,
        configuring the skeleton, and preparing visualization and segmentation tools.

        Args:
            model_type (str): Identifer of the METRABS model to download, uses metrabs mob3l y4t as default
            skeleton (str): Skeleton layout for pose detection, uses smpl+head_30 as default
        """
        self.model_path = self.download_model(model_type)
        self.model = tf.saved_model.load(self.model_path)
        self.skeleton = skeleton
        self.joint_names = self.model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        self.joint_edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
        self.frame_count = 0

        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        self.region_colors = {
            "left_arm": "#1f77b4",  # blue
            "right_arm": "#ff7f0e",  # orange
            "left_leg": "#2ca02c",  # green
            "right_leg": "#9467bd",  # purple
            "torso": "#8c564b",  # brown
            "head": "#d62728"  # dark red
        }

        self.joint_region_map = {
            "lsho_smpl": "left_arm", "lelb_smpl": "left_arm", "lwri_smpl": "left_arm", "lhan_smpl": "left_arm",
            "rsho_smpl": "right_arm", "relb_smpl": "right_arm", "rwri_smpl": "right_arm", "rhan_smpl": "right_arm",
            "lhip_smpl": "left_leg", "lkne_smpl": "left_leg", "lank_smpl": "left_leg", "ltoe_smpl": "left_leg",
            "rhip_smpl": "right_leg", "rkne_smpl": "right_leg", "rank_smpl": "right_leg", "rtoe_smpl": "right_leg",
            "pelv_smpl": "torso", "bell_smpl": "torso", "spin_smpl": "torso", "thor_smpl": "torso", "neck_smpl": "torso",
            "head_smpl": "head", "htop_mpi_inf_3dhp": "head", "learcoco": "head",
        }

        self.ground_joints = {"lank_smpl", "rank_smpl", "ltoe_smpl", "rtoe_smpl"}



    def download_model(self, model_type):
        """
        Downloads and extracts the AI model for the skeleton's pose detection.

        Args:
            model_type (str): Downloads the specified model type.
        Returns:
            str: The path of the recently saved model.
        """
        server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'

        model_zip_path = tf.keras.utils.get_file(
            origin=f'{server_prefix}/{model_type}_20211019.zip',
            cache_subdir='models',
            extract=False)

        model_extract_path = os.path.join(os.path.dirname(model_zip_path), model_type)

        if not os.path.exists(model_extract_path):
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_zip_path))

            extracted_folder = os.path.join(os.path.dirname(model_zip_path), f"{model_type}_20211019")
            if os.path.exists(extracted_folder):
                os.rename(extracted_folder, model_extract_path)

        return model_extract_path


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
        input_image = tf.expand_dims(input_image, axis=0)

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



    def is_inside_body(self, x, y, mask, kernel=8, thresh=0.25):
        """
        Determines whether a 2D joint lies within the skater's body mask.

        Args:
            x (int): X coordinate of the joint.
            y (int): Y coordinate of the joint.
            mask (np.ndarray): Binary segmentation mask of the player.
            kernel (int): Radius of the neighborhood to sample.
            thresh (float): Minimum mean mask value to consider the joint inside the body.
        Returns:
             bool: True if the detection belongs to the body, false if not.
        """
        h, w = mask.shape
        y1, y2 = max(0, y - kernel), min(h, y + kernel)
        x1, x2 = max(0, x - kernel), min(w, x + kernel)
        area = mask[y1:y2, x1:x2]
        return np.mean(area) > thresh


    def draw_results(self, video_frames, player_detections):
        """
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

        self.frame_count = 0

        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)

        #Joint's out of the skater's body errors are saved in a log file
        log_file = "joint_errors_log.txt"
        log_f = open(log_file, "w", buffering=1, encoding="utf-8")
        log_f.write("Frame,Joint,X,Y,Status\n")

        joints = self.joint_names if hasattr(self, "joint_names") else [str(i) for i in range(24)]


        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]

            for track_id, values in frame_dict.items():

                #Setup for the frame's structure
                fig = plt.figure(figsize=(14, 7))
                gs = fig.add_gridspec(
                    2, 2,
                    width_ratios=[2.3, 1],
                    height_ratios=[1, 1]
                )

                image_ax = fig.add_subplot(gs[:, 0])
                pose_ax = fig.add_subplot(gs[0, 1], projection="3d")
                graph_ax = fig.add_subplot(gs[1, 1])

                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)
                h, w = image.shape[:2]

                image_ax.set_xlim(0, w)
                image_ax.set_ylim(h, 0)

                image_ax.set_xlabel("X (px)")
                image_ax.set_ylabel("Y (px)")
                image_ax.tick_params(axis='both', labelsize=8)

                raw_bbox = values.get("boxes", None)
                raw_pose2d = values.get("poses2d", None)
                raw_pose3d = values.get("poses3d", None)

                bbox = None
                pose2d = None
                pose3d = None

                if raw_bbox is not None:
                    bbox = BBox(*raw_bbox)
                    image_ax.add_patch(
                        Rectangle((bbox.x, bbox.y), bbox.w, bbox.h,
                                  fill=False, color="yellow", linewidth=2)
                    )

                if raw_pose2d is not None:
                    pose2d = [Point2D(int(x), int(y)) for x, y in raw_pose2d.numpy()]

                if raw_pose3d is not None:
                    p3d = raw_pose3d.numpy()
                    p3d[..., 1], p3d[..., 2] = p3d[..., 2], -p3d[..., 1]
                    pose3d = [Point3D(x, y, z) for x, y, z in p3d]

                if pose2d and pose3d:
                    pose_ax.view_init(5, -75)
                    pose_ax.set_xlim3d(-1500, 1500)
                    pose_ax.set_zlim3d(-1500, 1500)
                    pose_ax.set_ylim3d(2000, 5000)
                    pose_ax.set_title("Pose 3D")

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.segmenter.process(rgb_frame)
                    mask = results.segmentation_mask > 0.3

                    joint_status = {}
                    ground_ys = []

                    for joint_idx, pt in enumerate(pose2d):
                        joint_name = joints[joint_idx] if joint_idx < len(joints) else None
                        if joint_name in self.ground_joints:
                            ground_ys.append(pt.y)

                    ground_y = max(ground_ys) if ground_ys else None

                    for joint_idx, pt in enumerate(pose2d):
                        joint_name = joints[joint_idx] if joint_idx < len(joints) else f"joint_{joint_idx}"
                        region = self.joint_region_map.get(joint_name, "torso")
                        base_color = self.region_colors[region]

                        #Error description for each point out of the bbox bounds
                        if bbox is not None:
                            if not (bbox.x <= pt.x <= bbox.x + bbox.w and
                                    bbox.y <= pt.y <= bbox.y + bbox.h):
                                errors_out_of_bbox.append({
                                    "frame": self.frame_count,
                                    "joint": joint_name,
                                    "x": pt.x,
                                    "y": pt.y,
                                    "bbox": f"({bbox.x:.1f},{bbox.y:.1f})-({bbox.x + bbox.w:.1f},{bbox.y + bbox.h:.1f})"
                                })

                        #Selects the color of the body part depending on if it's inside the skater's body
                        if 0 <= pt.y < mask.shape[0] and 0 <= pt.x < mask.shape[1]:
                            inside = self.is_inside_body(pt.x, pt.y, mask, kernel=13, thresh=0.10)
                            color = base_color if inside else "red"
                            status = "OK" if inside else "OUTSIDE_BODY"
                        else:
                            color = "gray"
                            status = "OUT_OF_BOUNDS"

                        joint_status[joint_idx] = {
                            "pt": pt,
                            "name": joint_name,
                            "color": color,
                            "status": status
                        }


                        log_f.write(f"{self.frame_count},{joint_name},{pt.x},{pt.y},{status}\n")

                        #Normalize Y coordinate relative to ground to keep graphs consistent across frames
                        if ground_y is not None:
                            y_norm = ground_y - pt.y
                        else:
                            y_norm = 0

                        joints_study_data.append({
                            "frame": self.frame_count,
                            "joint": joint_idx,
                            "y_norm": y_norm,
                            "track_id": track_id
                        })

                    #Draws the skater's bones on the 2D frame and on the 3d panel and colors them
                    for i_start, i_end in self.joint_edges:
                        start = joint_status[i_start]
                        end = joint_status[i_end]

                        region = self.joint_region_map.get(start["name"], "torso")
                        base_color = self.region_colors[region]
                        color_line = "red" if ("red" in (start["color"], end["color"])) else base_color

                        image_ax.plot(
                            [start["pt"].x, end["pt"].x],
                            [start["pt"].y, end["pt"].y],
                            color=color_line,
                            linewidth=1.8
                        )

                        pose_ax.plot(
                            [pose3d[i_start].x, pose3d[i_end].x],
                            [pose3d[i_start].y, pose3d[i_end].y],
                            [pose3d[i_start].z, pose3d[i_end].z],
                            color=color_line
                        )

                    #Draws the skarter's joints on the frame and on the 3d panel
                    for j in joint_status.values():
                        image_ax.scatter(
                            j["pt"].x, j["pt"].y,
                            s=22,
                            c=j["color"],
                            edgecolors="none",
                            zorder=4
                        )

                    pose_ax.scatter(
                        [p.x for p in pose3d],
                        [p.y for p in pose3d],
                        [p.z for p in pose3d],
                        s=5
                    )

                    df = pd.DataFrame(joints_study_data)

                    graph_ax.clear()
                    graph_ax.set_title("Evolución Y de articulaciones")
                    graph_ax.set_xlabel("Frame")
                    graph_ax.set_ylabel("Y")

                    #Black line to stablish where the ground is in the graph
                    graph_ax.axhline(0, color="black", linestyle="--", linewidth=1)

                    for joint_idx in df["joint"].unique():
                        df_joint = df[df["joint"] == joint_idx]
                        graph_ax.plot(
                            df_joint["frame"],
                            df_joint["y_norm"],
                            alpha=0.6,
                            linewidth=1
                        )

                    graph_ax.set_xlim(0, self.frame_count + 1)


                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

                frame_path = os.path.join(output_dir, f"frame_{self.frame_count:04d}.jpg")
                cv2.imwrite(frame_path, mat_frame)
                output_video_frames.append(mat_frame)

                plt.close(fig)

        log_f.close()

        #Log for saving the every point incorrectly place out of the bounding box
        if errors_out_of_bbox:
            csv_path = "errors_out_of_bbox.csv"
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "joint", "x", "y", "bbox"])
                writer.writeheader()
                writer.writerows(errors_out_of_bbox)

            print(f"📄 CSV guardado con {len(errors_out_of_bbox)} errores fuera de la bbox")

        print(f"Frames guardados en: {output_dir}")
        print(f"Log articulaciones: {log_file}")

        return output_video_frames

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


