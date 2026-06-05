# -*- coding: utf-8 -*-
"""
read videos

@author: pc
"""

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

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #MeTRAbs model rotates each frame 90 degrees counterclockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frames.append(rotated_frame)
    cap.release()
    return frames

"""
def rotate_batch_worker(batch):
    return [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in batch]


def read_video_batch(video_path, batch_size=32):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error al abrir el video.")
        return []

    frames_batch = []
    futures = []

    with ProcessPoolExecutor(max_workers=3) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_batch.append(frame)

            if len(frames_batch) == batch_size:
                futures.append(executor.submit(rotate_batch_worker, frames_batch))
                frames_batch = []

        if frames_batch:
            futures.append(executor.submit(rotate_batch_worker, frames_batch))

        cap.release()

        rotated_frames = []
        for f in futures:
            rotated_frames.extend(f.result())

    return rotated_frames
"""


def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        print("No hay frames para guardar.")
        return

    frame_height, frame_width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Video guardado exitosamente en: {output_video_path}")

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
"""