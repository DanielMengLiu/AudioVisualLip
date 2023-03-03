# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
""" This code was adapted from 
@InProceedings{martinez2020lipreading,
  author       = "Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja",
  title        = "Lipreading using Temporal Convolutional Networks",
  booktitle    = "ICASSP",
  year         = "2020",
} by Meng Liu """

""" face landmark detection for lip biometrics"""

import os
import cv2
import face_alignment
import argparse

import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

num_workers = 12
gpus = [2,4,6,7]

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lip biometrics Pre-processing -- 1. face detection')
    parser.add_argument('--video-dir', default='/datasets2/mobilelip/video', help='video directory')
    parser.add_argument('--landmark-dir', default='/datasets2/mobilelip/landmark', help='landmark directory')
    parser.add_argument('--manifest', default='preprocess/mobilelip/filelist', help='the manifest for landmark detection') #/data/liumeng/SyncLip2/preprocess/lrs3/lrs3_landmark.failure
  
    args = parser.parse_args()
    return args

args = load_args()

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

def run(gpu, iStart, iEnd, files, v_list, queue=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    
    for fl in range(iStart,iEnd):
        file = os.path.join(args.video_dir, files[fl]+'.mp4')
        video_crop = extract_opencv(file)
        if len(video_crop) == 0:
            v_list.append(file)
            logging.warning('empty file')
            continue
        
        session = files[fl].replace('/'+files[fl].split('/')[-1],'')
        if (not os.path.exists(os.path.join(args.landmark_dir,session))):
            os.makedirs(os.path.join(args.landmark_dir,session))

        list_landmarks = []
        path_landmark_crop = os.path.join(args.landmark_dir,files[fl]+'.npz')
        if (os.path.exists(path_landmark_crop)):
            continue
        for j in range(0,len(video_crop)):
            try:
                landmarks = fa.get_landmarks(video_crop[j])
                dic_landmarks = [{'facial_landmarks':landmarks[0]}]
                list_landmarks.append(dic_landmarks) 
            except Exception as e:
                logging.warning(e)
                print(file)
                v_list.append(file)
                if j == len(video_crop)-1:
                    list_landmarks.append([x for x in list_landmarks if x][-1])
                else:
                    list_landmarks.append(None)
                continue
        np.savez_compressed(path_landmark_crop, data=list_landmarks)

            
def extract_lip(manifest):
    abnorm_list = list()
    reader = open(manifest,'r')
    filelist = reader.read().splitlines()
    # if 'failure' not in manifest:
    #     filelist = [x.split(',')[2] for x in filelist]
    v_list = Manager().list()
    queues = [Queue(ctx=multiprocessing.get_context()) for i in range(num_workers)]

    part = list(range(0, len(filelist)+1, int(len(filelist)//num_workers)))
    part[-1] = len(filelist)
    
    args = [(gpus[i%len(gpus)], part[i], part[i+1], filelist, v_list, queues[i]) for i in range(num_workers)]

    jobs = [Process(target=run, args=(a)) for a in args]
    for j in jobs: j.start()
    for j in jobs: j.join()        
    
    for fn in v_list:
        abnorm_list.append(fn)
    if 'failure' not in manifest:
        with open('preprocess/mobilelip/mobilelip_landmark.failure','a+') as w:
            for fn in abnorm_list:
                w.writelines(fn+'\n')
           
if(__name__ == '__main__'):
    extract_lip(args.manifest)
 