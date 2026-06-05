# -*- coding: utf-8 -*-
"""
Padel analyzer

@author: pc
"""
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



from utils import (read_video,
                   save_video)

from trackers import PlayerTracker

def main():

    #Read video
    input_video = "demo.mp4"

    video_frames = read_video(input_video)

    #Player Tracker's Initialization
    player_tracker = PlayerTracker(model_type='metrabs_mob3l_y4t')
    #Better model but also slower:
    #player_tracker = PlayerTracker(model_type='metrabs_eff2l_y4')


    #Pose detection
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )

    #Player's training analysis
    output_video_frames = player_tracker.draw_results(video_frames, player_detections)


    #Output video
    save_video(output_video_frames, "output_videos/result_prueba_video_nuevo.avi")


if __name__ == '__main__':
    main()