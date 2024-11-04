# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:30:44 2024

@author: pc
"""


import cv2
import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

class PlayerTracker:
    def __init__(self,model_path_or_url = 'https://bit.ly/metrabs_l', skeleton = 'smpl+head_30'):
        self.model = tfhub.load(model_path_or_url)
        self.skeleton = skeleton
        self.joint_names = self.model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        self.joint_edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
        self.frame_count = 0
       
    def detect_frames(self,frames, read_from_stub=False, stub_path=None, n = 2):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame,n)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame, n = 2):
        results = self.model.detect_poses(frame,max_detections = 2, skeleton=self.skeleton)

        player_dict = {}
        self.frame_count +=1 
        print(self.frame_count)
        player_dict[self.frame_count] = {}
        track_id = 0
        for boxes, poses2d, poses3d in zip(results['boxes'], results['poses2d'], results['poses3d']):
            track_id += 1
            player_dict[self.frame_count][track_id] = {'boxes': boxes,
                                      'poses2d': poses2d,
                                      'poses3d': poses3d}
            
        
        return player_dict
    #Modify it
    def draw_bboxes(self,video_frames, player_detections):
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
                
                #Values
                
                bbox = values.get('boxes',None)
                pose2d = values.get('poses2d',None)
                pose3d = values.get('poses3d', None)
                # Draw Bounding Boxes
                x, y, w, h, c = bbox
                image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
                #Draw pose detections
                pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
                pose_ax.view_init(5, -75)
                pose_ax.set_xlim3d(-1500, 1500)
                pose_ax.set_zlim3d(-1500, 1500)
                pose_ax.set_ylim3d(2000, 5000)
                if not None in (bbox, pose2d, pose3d):
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