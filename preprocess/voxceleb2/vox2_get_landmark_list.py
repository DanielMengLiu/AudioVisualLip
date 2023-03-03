import numpy as np
import argparse
from glob import glob
import os

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--landmark-direc', default='/datasets3/voxceleb2/landmark/dev/', help='landmark directory')
    parser.add_argument('--landmark-list', default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/preprocessing/voxceleb2/landmarklist', help='mean face pathname')

    args = parser.parse_args()
    return args

args = load_args()

pattern = args.landmark_direc + "/*/*/*.npz"
filelist = sorted(glob(pattern))

with open(args.landmark_list, 'w') as w:
    for filepath in filelist:
        no = filepath.split('.')[0]
        spk = no.split('/')[-3]
        session = no.split('/')[-2]
        fn = no.split('/')[-1]
        w.write(os.path.join(spk,session,fn)+'\n')
w.close()
