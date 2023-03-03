import os
import numpy as np
import argparse
from glob import glob
import random

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing -- 2. get mean face')
    # -- utils
    parser.add_argument('--landmark-dir', default='/datasets2/voxceleb2/landmark', help='landmark directory')
    parser.add_argument('--mean-face', default='/data/liumeng/SyncLip3/preprocess/voxceleb2/vox2_mean_face_100k.npy', help='mean face pathname')
    parser.add_argument('--mean-num', default=100000, help='number of faces are used for generating the mean face')

    args = parser.parse_args()
    return args

args = load_args()

pattern = os.path.join(args.landmark_dir, 'dev', "*/*/*.npz")
filelist = glob(pattern)

background_face = np.zeros((68,2), dtype = float)
for i in range(0, args.mean_num):
    index = random.randint(0,len(filelist)-1)
    filename = filelist[index]
    frame_block = np.load(filename, allow_pickle=True)['data']
    bf = np.zeros((68,2), dtype = float)
    num_frames = len(frame_block)
    for j in range(0, len(frame_block)):
        if frame_block[j] is not None:
            bf += frame_block[j][0]['facial_landmarks']
        else:
            num_frames -= 1
    mf = bf / num_frames
    background_face += mf
    if i % 100 == 0:
        print(str(i/1000)+'k have been done')
        
mean_face = background_face / args.mean_num
np.save(args.mean_face, mean_face)
print('all done')

