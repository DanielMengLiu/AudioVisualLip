import numpy as np
import argparse
from glob import glob
import os

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default='/data/datasets/TCD-TIMIT/volunteers', help='video directory')
    parser.add_argument('--video-list', default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/preprocessing/videolist', help='video list')

    args = parser.parse_args()
    return args

args = load_args()

pattern = args.video_direc + "/*/straightcam/*.mp4"
filelist = sorted(glob(pattern))

with open(args.video_list, 'w') as w:
    for filepath in filelist:
        basename = filepath.split('/')[-1]
        dirname =  filepath.replace(basename, '')
        filename = basename.split('_')[0]

w.close()
