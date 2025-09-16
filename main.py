# -*- coding: utf-8 -*-
"""
Padel analyzer

@author: pc
"""

from utils import (read_video_batch,
                   save_video_batch)


from trackers import PlayerTracker

def main():
    #Read video
    input_video = "demo.mp4"
    video_frames = read_video_batch(input_video)

    #Detect players
    player_tracker = PlayerTracker(model_type='metrabs_mob3l_y4t')


    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl",
                                                     n = 1
                                                     )



    #Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    #Output video
    save_video_batch(output_video_frames, "output_videos/result_parallelized.avi")

if __name__ == '__main__':
    main()