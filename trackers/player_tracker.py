# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:30:44 2024

@author: pc
"""
import os

import zipfile

import cv2
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class PlayerTracker:
    def __init__(self,model_type='metrabs_mob3l_y4t', skeleton = 'smpl+head_30'):
        self.model_path = self.download_model(model_type)
        self.model = tf.saved_model.load(self.model_path)
        self.skeleton = skeleton
        self.joint_names = self.model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        self.joint_edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
        self.frame_count = 0


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


    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        self.frame_count = 0
        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]
            for track_id, values in frame_dict.items():
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                image_ax.imshow(image)

                # Values

                bbox = values.get('boxes', None)
                pose2d = values.get('poses2d', None)
                pose3d = values.get('poses3d', None)

                # Draw Bounding Boxes
                if bbox is not None:
                    x, y, w, h, c = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

                # Draw pose detections
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

            fig.canvas.draw()

            img_plot = np.array(fig.canvas.renderer.buffer_rgba())

            mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

            output_video_frames.append(mat_frame)

            plt.close()

        return output_video_frames