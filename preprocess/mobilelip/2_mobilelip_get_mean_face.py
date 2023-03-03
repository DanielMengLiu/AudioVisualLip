import os
import numpy as np
import argparse
from glob import glob
import random

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing -- 2. get mean face')
    # -- utils
    parser.add_argument('--landmark-dir', default='/datasets2/mobilelip/landmark', help='landmark directory')
    parser.add_argument('--mean-face', default='/data/linchenghan/lip_inter_dataset/mobilelip/mobilelip_mean_face_3000.npy', help='mean face pathname')
    parser.add_argument('--mean-num', default=3000, help='number of faces are used for generating the mean face')

    args = parser.parse_args()
    return args

args = load_args()

pattern = os.path.join(args.landmark_dir, "*/*/*.npz")
filelist = glob(pattern)

background_face = np.zeros((68,2), dtype = float)
for i in range(0, args.mean_num):
    index = random.randint(0,len(filelist)-1)
    filename = filelist[index]
    frame_block = np.load(filename, allow_pickle=True)['data']
    bf = np.zeros((68,2), dtype = float)
    len_frame_block = len(frame_block)
    for j in range(0, len_frame_block):
        if frame_block[j] is not None:
            bf += frame_block[j][0]['facial_landmarks']
        else:
            len_frame_block -= 1
    mf = bf / len_frame_block
    background_face += mf

mean_face = background_face / args.mean_num
if not os.path.exists(args.mean_face):
    os.makedirs(args.mean_face.replace(args.mean_face.split('/')[-1], ''), exist_ok=True)
np.save(args.mean_face, mean_face)
print('done')

