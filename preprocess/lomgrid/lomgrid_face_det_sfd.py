import os
import cv2
import face_alignment
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue
import logging
import numpy as np

# import sys
# import dlib
# import time
# import argparse
# import csv
# import soundfile as sf
# from python_speech_features import mfcc

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

num_workers = 14
gpus = [0,1,2,3,4,5,6,7]
video_dir = '/data/datasets/Lombard GRID/lombardgrid/front'
landmark_dir = '/datasets2/lomgrid/landmark/'
manifest = '/data/liumeng/SyncLip/data/manifest/manifest_lombardgrid_test.txt'


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
        file = os.path.join(video_dir, files[fl].split('/')[1]+'.mov')
        try:
            video = extract_opencv(file)
            if len(video) == 0:
                v_list.append(file)
                logging.warning('empty file')
                continue
            
            spk = file.split('/')[-3]
            session = file.split('/')[-2]
            if (not os.path.exists(os.path.join(landmark_dir,spk,session))):
                os.makedirs(os.path.join(landmark_dir,spk,session))

            list_landmarks = []
            video_crop = video
            path_landmark_crop = file.replace(video_dir,landmark_dir).split('.')[0]
            if (os.path.exists(path_landmark_crop+'.npz')):
                continue
            for j in range(0,len(video)):
                landmarks = fa.get_landmarks(video_crop[j])
                dic_landmarks = [{'facial_landmarks':landmarks[0]}]
                list_landmarks.append(dic_landmarks)       
            np.savez(path_landmark_crop, data=list_landmarks)
        except Exception as e:
            logging.warning(e)
            print(file)
            v_list.append(file)
            
def extract_lip():
    abnorm_list = list()
    reader = open(manifest,'r')
    filelist = reader.read().splitlines()
    
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
    with open('lomgrid_landmark_failure.log','w+') as w:
        for fn in abnorm_list:
            w.writelines(fn+'\n')
           
if(__name__ == '__main__'):
    extract_lip()
 