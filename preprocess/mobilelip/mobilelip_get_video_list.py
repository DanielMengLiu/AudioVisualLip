import numpy as np
import argparse
from glob import glob
import os

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipbiometrics Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default='/datasets2/mobilelip/video', help='directory')
    parser.add_argument('--video-list', default='/data/liumeng/Short-Short/preprocess/mobilelip/filelist', help='list')

    args = parser.parse_args()
    return args

args = load_args()

pattern = args.video_direc + "/*/*/*.mp4"
filelist = sorted(glob(pattern))

with open(args.video_list, 'w') as w:
    for filepath in filelist:
        no = filepath.split('.')[0]
        spk = no.split('/')[-3]
        session = no.split('/')[-2]
        fn = no.split('/')[-1]
        w.write(os.path.join(spk,session,fn)+'\n')
w.close()
