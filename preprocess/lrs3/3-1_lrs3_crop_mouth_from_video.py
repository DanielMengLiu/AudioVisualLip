#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
""" This code was adapted from 
@InProceedings{martinez2020lipreading,
  author       = "Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja",
  title        = "Lipreading using Temporal Convolutional Networks",
  booktitle    = "ICASSP",
  year         = "2020",
} by Meng Liu"""

""" Crop Face ROIs from videos for face biometrics"""
import os
import cv2
import argparse
import numpy as np
from collections import deque
import sys 
sys.path.append("..") 
from utils import *
from transform import *
from multiprocessing import Pool, Process, Queue
from skimage import io

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Face biometrics Pre-processing -- 3-1. crop face')
    # -- utils
    parser.add_argument('--video-dir', default='/datasets1/LRS3/video', help='video directory')
    parser.add_argument('--landmark-dir', default='/datasets1/LRS3/landmark', help='landmark directory')
    parser.add_argument('--lip-dir', default='/datasets1/LRS3/face', help='the directory of saving face ROIs')
    parser.add_argument('--manifest', default='/data/liumeng/SyncLip2/data/manifest/LRS3_test_manifest.csv', help='list of detected video and its subject ID')

    # -- mean face utils
    parser.add_argument('--mean-face', default='/data/liumeng/SyncLip2/preprocess/lrs3/lrs3_mean_face_30000.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=192, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=192, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=1, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=True, action='store_true', help='convert2grayscale')
    parser.add_argument('--save-type', default='.npz', help='npz or jpg')
    args = parser.parse_args()
    return args

args = load_args()

# -- mean face utils
STD_SIZE = (224, 224) # resolution
mean_face_landmarks = np.load(args.mean_face)
stablePntsIDs = [33, 36, 39, 42, 45]

def crop_patch(video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    frame_idx = 0 #
    frame_gen = read_video(video_pathname)
    
    q_frame, q_landmarks = deque(), deque()
    sequence = []
        
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        
        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
    
        if len(q_frame) == args.window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                        mean_face_landmarks[stablePntsIDs, :],
                                        cur_frame,
                                        STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch( trans_frame,
                                        trans_landmarks[args.start_idx:args.stop_idx],
                                        args.crop_height//2,
                                        args.crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[args.start_idx:args.stop_idx],
                                            args.crop_height//2,
                                            args.crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]


def run(files):
    for filename_idx, filename in enumerate(files):
        video_path = os.path.join(args.video_dir, filename+'.mp4')
        # video_path = '/datasets1/LRS3/video/trainval/mwzBJF5Q7Vw/50008.mp4'
        landmark_path = os.path.join(args.landmark_dir, filename+'.npz')
        # landmark_path = '/datasets1/LRS3/landmark/trainval/mwzBJF5Q7Vw/50008.npz'

        if args.save_type == '.jpg':
            lip_path = os.path.join(args.lip_dir, filename+'.jpg')
        elif args.save_type == '.npz':
            lip_path = os.path.join(args.lip_dir, filename+'.npz')
        # lip_path = '/datasets1/LRS3/face/trainval/mwzBJF5Q7Vw/50008.npz'
        if os.path.exists(lip_path):
            continue

        multi_sub_landmarks = np.load(landmark_path, allow_pickle=True)['data']

        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                if multi_sub_landmarks[frame_idx]:
                    landmarks[frame_idx] = multi_sub_landmarks[frame_idx][0]['facial_landmarks']
            except IndexError:
                continue
        
        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # -- crop
        sequence = crop_patch(video_path, preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
        data_resize = []
        # for di in range(0, len(data)):
        #     data_resize.append(cv2.resize(data[di], (96,96), interpolation=cv2.INTER_CUBIC))
        if args.save_type == '.jpg':
            save2jpg(lip_path, data=data)
        elif args.save_type == '.npz':
            save2npz(lip_path, data=data)

if(__name__ == '__main__'):
    lines = open(args.manifest).read().splitlines()
    data = [x.split(',')[2] for x in lines]
    processes = []
    n_p = 40
    bs = len(data) // n_p
    for i in range(n_p):
        if(i == n_p - 1):
            bs = len(data)
        p = Process(target=run, args=(data[:bs],))
        data = data[bs:]
        p.start()
        processes.append(p)
    assert(len(data) == 0)
    for p in processes:
        p.join()

    print('Done.')
