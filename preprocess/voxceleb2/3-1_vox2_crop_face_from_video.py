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

""" Crop Face ROIs (high resolution 196x196) from videos for face biometrics"""
import os
import cv2
import argparse
import numpy as np
import sys 
sys.path.append("..") 
from utils import *
from transform import *
from multiprocessing import Process

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Face biometrics Pre-processing -- 3-1. crop face')
    # -- utils
    parser.add_argument('--video-dir', default='/datasets2/voxceleb2/video/dev/mp4', help='video directory')
    parser.add_argument('--lip-dir', default='/datasets3/voxceleb2/face', help='the directory of saving face ROIs')
    parser.add_argument('--manifest', default='data/manifest/voxceleb2_dev_manifest.csv', help='list of detected video and its subject ID')

    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=224, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=224, type=int, help='the height of mouth ROIs')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=False, action='store_true', help='convert2grayscale')
    parser.add_argument('--save-type', default='.jpg', help='npz or jpg')
    args = parser.parse_args()
    return args

args = load_args()

def crop_patch(video_pathname):

    """Crop face patch
    :param str video_pathname: pathname for the video_dieo
    """
    frame_gen = read_video(video_pathname)
    sequence = []
        
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        
        sequence.append(frame)

    data = [x for i, x in enumerate(sequence) if i%12==0]  

    return np.array(data)

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

        if args.save_type == '.jpg':
            lip_path = os.path.join(args.lip_dir, filename+'.jpg')
        elif args.save_type == '.npz':
            lip_path = os.path.join(args.lip_dir, filename+'.npz')
        # lip_path = '/datasets1/LRS3/face/trainval/mwzBJF5Q7Vw/50008.npz'
        if not os.path.exists(video_path):
            print('video does not exist')
            continue
        
        # -- crop
        sequence = crop_patch(video_path)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if args.convert_gray else sequence
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
