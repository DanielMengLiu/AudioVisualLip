#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" Crop Mouth ROIs from videos for lipreading"""

import os
import cv2
import glob
import argparse
import numpy as np
from collections import deque

from tqdm import tqdm

from utils import *
from transform import *
from multiprocessing import Pool, Process, Queue, Manager
import face_alignment
from skimage import io
import time

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lombardgrid Pre-processing')
    # -- utils
    parser.add_argument('--video-dir', default='/data/datasets/Lombard GRID/lombardgrid/front', help='raw video directory')
    parser.add_argument('--landmark-dir', default='/datasets2/lomgrid/landmark/', help='landmark directory')
    parser.add_argument('--lip-dir', default='/data/linchenghan/lip_inter_dataset/lomgrid/lip', help='the directory of saving mouth ROIs')
    parser.add_argument('--filename-path', default='/data/liumeng/SyncLip/data/manifest/manifest_lombardgrid_test.txt', help='list of detected video and its subject ID')
    parser.add_argument('--failure-file',
                        default='/data/linchenghan/lip_inter_dataset/lomgrid/lip/lomgrid_lip.failure',
                        help='failure description file')

    # -- mean face utils
    parser.add_argument('--mean-face', default='/data/liumeng/SyncLip/preprocess/voxceleb2/lombardgrid_2000_mean_face.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=124, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=124, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=True, action='store_true', help='convert2grayscale')
    parser.add_argument('--save-type', default='.mp4', help='mp4,npz or jpg')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

# -- mean face utils
STD_SIZE = (720, 480) # resolution
mean_face_landmarks = np.load(args.mean_face)
stablePntsIDs = [33, 36, 39, 42, 45]


def crop_patch(video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    frame_idx = 0
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
    landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
    landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def run(files,failure_list):
    for filename in tqdm(files):
        video_path = os.path.join(args.video_dir, filename.split('/')[1]+'.mov')
        landmark_path = os.path.join(args.landmark_dir, filename.split('/')[1]+'.npz')
        lip_path = os.path.join(args.lip_dir, filename.split('/')[1] + args.save_type)

        if os.path.exists(lip_path):
            continue
        if not os.path.exists(landmark_path):
            print('landmark not exists: ' + landmark_path)
            failure_list.append(landmark_path)
            continue
        try:
            multi_sub_landmarks = np.load(landmark_path, allow_pickle=True)['data']
        except:
            print('broken landmark: ' + landmark_path)
            failure_list.append(landmark_path)
            continue

        landmarks = [None] * len( multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][0]['facial_landmarks']
            except IndexError:
                print('out of index in this landmark: ' + landmark_path)
                failure_list.append(landmark_path)
                continue
        
        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            print('cannot preprocess landmark: ' + landmark_path)
            failure_list.append(landmark_path)
            continue

        # -- crop
        sequence = crop_patch(video_path, preprocessed_landmarks)
        if sequence is None:
            print('cannot crop from video: ' + video_path)
            failure_list.append(video_path)

        # -- save
        data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
        data_resize = []
        for di in range(0, len(data)):
            data_resize.append(cv2.resize(data[di], (96,96), interpolation=cv2.INTER_CUBIC))
        try:
            if args.save_type == '.mp4':
                save2mp4(lip_path, data=data_resize, convert_gray=args.convert_gray)
            if args.save_type == '.jpg':
                save2jpg(lip_path, data=data_resize)
            elif args.save_type == '.npz':
                save2npz(lip_path, data=data_resize)
        except Exception as e:
            print('cannot save to'+ args.save_type +': ' + video_path)
            failure_list.append(video_path)



if(__name__ == '__main__'):

    lines = open(args.filename_path).read().splitlines()
    data = list(filter(lambda x: 'test' in x, lines)) if args.testset_only else lines

    processes = []
    n_p = 80
    bs = len(data) // n_p
    failure_list = Manager().list()
    for i in range(n_p):
        if(i == n_p - 1):
            bs = len(data)
        p = Process(target=run, args=(data[:bs],failure_list))
        data = data[bs:]
        p.start()
        processes.append(p)
    assert(len(data) == 0)
    for p in processes:
        p.join()

    failure_file = args.failure_file
    if not os.path.exists(failure_file):
        os.makedirs(failure_file.replace(failure_file.split('/')[-1], ''), exist_ok=True)
    with open(failure_file, 'w+') as w:
        for fn in failure_list:
            w.writelines(fn + '\n')

    print('Done.')
