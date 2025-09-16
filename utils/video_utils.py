# -*- coding: utf-8 -*-
"""
read videos

@author: pc
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import urllib.request

def get_video(source, temppath='/tmp/video.mp4'):
    if not source.startswith('http'):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath

#Original version, no parallelization
"""
def read_video(video_path):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"read_video took {elapsed_time:.4f} seconds.")
    return frames
"""


def read_video_batch(video_path, batch_size=16):

    def read_all_frames():
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(rotated_frame)
        cap.release()
        return frames

    all_frames = read_all_frames()

    def process_batch(batch):
        return batch

    frame_batches = [all_frames[i:i + batch_size] for i in range(0, len(all_frames), batch_size)]

    rotated_frames = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch) for batch in frame_batches]
        for f in futures:
            rotated_frames.extend(f.result())

    return rotated_frames

#Original version, no parallelization
"""
def save_video(output_video_frames, output_video_path):
    start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"save_video took {elapsed_time: .4f} seconds.")
    out.release()

"""


def save_video_batch(output_video_frames, output_video_path, batch_size=16):

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_height, frame_width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))

    def process_batch(batch):
        return [frame for frame in batch]

    batches = [output_video_frames[i:i + batch_size] for i in range(0, len(output_video_frames), batch_size)]


    with ThreadPoolExecutor() as executor:
        processed_batches = list(executor.map(process_batch, batches))

    for batch in processed_batches:
        for frame in batch:
            out.write(frame)


    out.release()
