# -*- coding: utf-8 -*-
"""
Padel analyzer

@author: pc
"""
from utils import (read_video, 
                   save_video)

import tensorflow as tf
import tensorflow_hub as tfhub


from trackers import PlayerTracker

def main():
    #Read video
    input_video = "demo.mp4"
    video_frames = read_video(input_video)

    #Detect players
    player_tracker = PlayerTracker(model_path_or_url = 'https://bit.ly/metrabs_l')
    
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl",
                                                     n = 1
                                                     )
    
    #Output video
    
    #Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    
    save_video(output_video_frames, "output_videos/result.avi")
    
if __name__ == '__main__':
    main()