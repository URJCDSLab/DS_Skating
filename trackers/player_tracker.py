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

Point2D = namedtuple("Point2D", ["x", "y"])
Point3D = namedtuple("Point3D", ["x", "y", "z"])
BBox = namedtuple("BBox", ["x", "y", "w", "h", "conf"])

class PlayerTracker:
    def __init__(self,model_type='metrabs_mob3l_y4t', skeleton = 'smpl+head_30'):
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
        self.region_map = {
            "left_arm": ["lsho_smpl", "lelb_smpl", "lwri_smpl", "lhan_smpl"],
            "right_arm": ["rsho_smpl", "relb_smpl", "rwri_smpl", "rhan_smpl"],
            "left_leg": ["lhip_smpl", "lkne_smpl", "lank_smpl", "ltoe_smpl"],
            "right_leg": ["rhip_smpl", "rkne_smpl", "rank_smpl", "rtoe_smpl"],
            "torso": ["pelv_smpl", "bell_smpl", "spin_smpl", "thor_smpl", "neck_smpl"],
            "head": ["head_smpl", "htop_mpi_inf_3dhp", "learcoco"],
        }


    def download_model(self, model_type):
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

    def detect_frames(self, frames, read_from_stub=False, stub_path=None, n=2):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame, n)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)


        return player_detections

    def detect_frame(self,frame, n = 2):
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

    def get_joint_region(self, joint_name):
        for region, joint_list in self.region_map.items():
            if joint_name in joint_list:
                return region
        return "torso"

    def is_inside_body(self, x, y, mask, kernel=8, thresh=0.25):
        h, w = mask.shape
        y1, y2 = max(0, y - kernel), min(h, y + kernel)
        x1, x2 = max(0, x - kernel), min(w, x + kernel)
        area = mask[y1:y2, x1:x2]
        return np.mean(area) > thresh

    #Versión que pinta las extremidades de un color y de rojo las incorrectamente colocadas y con valores de los puntos y la bbox más visuales
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        self.frame_count = 0

        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, "joint_errors_log.txt")
        log_f = open(log_file, "w", buffering=1, encoding="utf-8")
        log_f.write("Frame,Joint,X,Y,Status\n")


        if hasattr(self, "joint_names"):
            joints = self.joint_names
        else:
            joints = [str(i) for i in range(24)]

        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]

            for track_id, values in frame_dict.items():
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)

                raw_bbox = values.get("boxes", None)
                raw_pose2d = values.get("poses2d", None)
                raw_pose3d = values.get("poses3d", None)

                bbox = None
                pose2d = None
                pose3d = None

                if raw_bbox is not None:
                    bbox = BBox(*raw_bbox)

                if raw_pose2d is not None:
                    pose2d = [Point2D(int(x), int(y)) for x, y in raw_pose2d.numpy()]

                if raw_pose3d is not None:
                    p3d = raw_pose3d.numpy()
                    p3d[..., 1], p3d[..., 2] = p3d[..., 2], -p3d[..., 1]
                    pose3d = [Point3D(x, y, z) for x, y, z in p3d]

                if bbox is not None:
                    image_ax.add_patch(
                        Rectangle((bbox.x, bbox.y), bbox.w, bbox.h, fill=False, color="yellow")
                    )

                if pose3d is not None and pose2d is not None:
                    pose_ax = fig.add_subplot(1, 2, 2, projection="3d")
                    pose_ax.view_init(5, -75)
                    pose_ax.set_xlim3d(-1500, 1500)
                    pose_ax.set_zlim3d(-1500, 1500)
                    pose_ax.set_ylim3d(2000, 5000)


                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.segmenter.process(rgb_frame)
                    mask = results.segmentation_mask > 0.3

                    joint_status = {}

                    for joint_idx, pt in enumerate(pose2d):
                        joint_name = joints[joint_idx] if joint_idx < len(joints) else f"joint_{joint_idx}"
                        region = self.get_joint_region(joint_name)
                        base_color = self.region_colors[region]

                        if 0 <= pt.y < mask.shape[0] and 0 <= pt.x < mask.shape[1]:
                            inside = self.is_inside_body(pt.x, pt.y, mask, kernel=13, thresh=0.10)
                            if inside:
                                status = "OK"
                                color = base_color
                            else:
                                status = "OUTSIDE_BODY"
                                color = "red"
                        else:
                            status = "OUT_OF_BOUNDS"
                            color = "gray"

                        joint_status[joint_idx] = {
                            "pt": pt,
                            "name": joint_name,
                            "color": color,
                            "status": status
                        }

                        log_f.write(f"{self.frame_count},{joint_name},{pt.x},{pt.y},{status}\n")

                    for i_start, i_end in self.joint_edges:
                        start = joint_status[i_start]
                        end = joint_status[i_end]

                        region = self.get_joint_region(start["name"])
                        base_color = self.region_colors[region]

                        color_line = "red" if (
                                start["color"] == "red" or end["color"] == "red"
                        ) else base_color

                        image_ax.plot(
                            [start["pt"].x, end["pt"].x],
                            [start["pt"].y, end["pt"].y],
                            color=color_line,
                            linewidth=1.8,
                            alpha=0.85
                        )

                        pose_ax.plot(
                            [pose3d[i_start].x, pose3d[i_end].x],
                            [pose3d[i_start].y, pose3d[i_end].y],
                            [pose3d[i_start].z, pose3d[i_end].z],
                            color=color_line
                        )

                    for j in joint_status.values():
                        image_ax.scatter(
                            j["pt"].x, j["pt"].y,
                            s=20,
                            c=j["color"],
                            edgecolors="none",
                            zorder=4
                        )

                    pose_ax.scatter(
                        [p.x for p in pose3d],
                        [p.y for p in pose3d],
                        [p.z for p in pose3d],
                        s=3
                    )

                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

                frame_path = os.path.join(output_dir, f"frame_{self.frame_count:04d}.jpg")
                cv2.imwrite(frame_path, mat_frame)
                output_video_frames.append(mat_frame)

                plt.close()

        log_f.close()

        print(f"Frames guardados en: {output_dir}")
        print(f"Log de articulaciones fuera del cuerpo: {log_file}")

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

#Versión atrasada que guarda los valores que salen de la bbox en un .csv
"""
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        self.frame_count = 0
        errores_fuera_bbox = []  

        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]

            for track_id, values in frame_dict.items():
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)

                bbox = values.get('boxes', None)
                pose2d = values.get('poses2d', None)
                pose3d = values.get('poses3d', None)

                if bbox is not None:
                    x, y, w, h, c = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

                if pose3d is not None and pose2d is not None:
                    pose3d = pose3d.numpy()
                    pose2d = pose2d.numpy()

                    if bbox is not None:
                        x_min, x_max = x, x + w
                        y_min, y_max = y, y + h
                        for joint_idx, (jx, jy) in enumerate(pose2d):
                            if not (x_min <= jx <= x_max and y_min <= jy <= y_max):
                                joint_name = self.joint_names[joint_idx] if joint_idx < len(
                                    self.joint_names) else f"Joint {joint_idx}"
                                errores_fuera_bbox.append({
                                    "frame": self.frame_count,
                                    "joint": joint_name,
                                    "x": round(float(jx), 2),
                                    "y": round(float(jy), 2),
                                    "bbox": f"({x_min:.1f},{y_min:.1f})-({x_max:.1f},{y_max:.1f})"
                                })

                    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
                    pose_ax.view_init(5, -75)
                    pose_ax.set_xlim3d(-1500, 1500)
                    pose_ax.set_zlim3d(-1500, 1500)
                    pose_ax.set_ylim3d(2000, 5000)
                    pose3d[..., 1], pose3d[..., 2] = pose3d[..., 2], -pose3d[..., 1]

                    for i_start, i_end in self.joint_edges:
                        image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
                        pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)

                    image_ax.scatter(*pose2d.T, s=2)
                    pose_ax.scatter(*pose3d.T, s=2)

                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                output_video_frames.append(mat_frame)
                plt.close(fig)

        if errores_fuera_bbox:
            csv_path = "errores_fuera_bbox.csv"
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "joint", "x", "y", "bbox"])
                writer.writeheader()
                writer.writerows(errores_fuera_bbox)
            print(f"\nArchivo guardado con {len(errores_fuera_bbox)} errores en '{csv_path}'")

        return output_video_frames
"""


#Versión atrasada con los gráficos X e Y de cada articulación
"""
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        self.frame_count = 0
        joints_study_data = []

        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]

            for track_id, values in frame_dict.items():
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)

                bbox = values.get('boxes', None)
                pose2d = values.get('poses2d', None)
                pose3d = values.get('poses3d', None)

                if bbox is not None:
                    x, y, w, h, c = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

                if pose3d is not None and pose2d is not None:
                    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
                    pose_ax.view_init(5, -75)
                    pose_ax.set_xlim3d(-1500, 1500)
                    pose_ax.set_zlim3d(-1500, 1500)
                    pose_ax.set_ylim3d(2000, 5000)

                    pose3d = pose3d.numpy()
                    pose2d = pose2d.numpy()
                    pose3d[..., 1], pose3d[..., 2] = pose3d[..., 2], -pose3d[..., 1]

                    for i_start, i_end in self.joint_edges:
                        image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
                        pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)

                    image_ax.scatter(*pose2d.T, s=2)
                    pose_ax.scatter(*pose3d.T, s=2)

                    num_joints = len(pose2d)
                    for joint_idx in range(num_joints):
                        x, y = pose2d[joint_idx]
                        joints_study_data.append({
                            "frame": self.frame_count,
                            "joint": joint_idx,
                            "x": x,
                            "y": y,
                            "track_id": track_id
                        })

                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                output_video_frames.append(mat_frame)
                plt.close(fig)

        df = pd.DataFrame(joints_study_data)

        plt.figure(figsize=(10, 6))
        for joint_idx in df["joint"].unique():
            df_joint = df[df["joint"] == joint_idx]

            if hasattr(self, "joint_names") and joint_idx < len(self.joint_names):
                joint_name = self.joint_names[joint_idx]
            else:
                joint_name = f"Joint {joint_idx}"

            plt.plot(df_joint["frame"], df_joint["y"], label=joint_name, alpha=0.5)

        plt.title("Movimiento Y de todas las articulaciones")
        plt.xlabel("Frame")
        plt.ylabel("Coordenada Y")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        return output_video_frames
        
"""

