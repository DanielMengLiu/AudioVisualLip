import numpy as np
import argparse
from glob import glob
import os
import csv
import random
from functools import reduce
import tqdm as tqdm

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='make trials')
    # -- utils
    parser.add_argument('--test-manifest', default='data/manifest/grid_test_manifest.csv', help='test utt list')
    parser.add_argument('--trial', default='data/trial/grid_E.txt', help='lrs3 test set trial')
  
    args = parser.parse_args()
    return args

args = load_args()
utts = []
with open(args.test_manifest, mode='r',newline='') as r:
    reader = csv.reader(r)
    for _,_,fn,_,_ in reader:
        utts.append(fn)
        
with open(args.trial, 'w') as w:
    for i in range(len(utts)):
        for j in range(len(utts)):
            if i < j:
                enroll_utt, eval_utt = utts[i], utts[j]
                enroll_spk, eval_spk = enroll_utt.split('/')[-2], eval_utt.split('/')[-2]
                label = int(enroll_spk == eval_spk)
                if random.randint(0,18024) % 18024 == 0:
                    w.write(str(label)+' '+enroll_utt+' '+eval_utt+'\n')

print('finished!')