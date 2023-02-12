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
    parser.add_argument('--test-manifest', default='data/manifest/lombardgrid_test_manifest.csv', help='test utt list')
    parser.add_argument('--trial', default='data/trial/lomgrid_E.txt', help='lrs3 test set trial')
  
    args = parser.parse_args()
    return args

args = load_args()
utts = []
with open(args.test_manifest, mode='r',newline='') as r:
    reader = csv.reader(r)
    for fn in reader:
        utts.append(fn)

cnt = 0
cnn = 0
with open(args.trial, 'w') as w:
    for i in range(len(utts)):
        for j in range(len(utts)):
            if i < j:
                enroll_utt, eval_utt = utts[i][0], utts[j][0]
                enroll_spk, eval_spk = enroll_utt.split('/')[-2], eval_utt.split('/')[-2]
                label = int(enroll_spk == eval_spk)
                if label == 1:
                    # if random.randint(1,10) % 10 == 0:
                    cnt += 1
                    print(cnt)
                    w.write(str(label)+' '+enroll_utt.split('/')[-1]+' '+eval_utt.split('/')[-1]+'\n')
    for i in range(len(utts)):
        for j in range(len(utts)):
            if i < j:
                enroll_utt, eval_utt = utts[i][0], utts[j][0]
                enroll_spk, eval_spk = enroll_utt.split('/')[-2], eval_utt.split('/')[-2]
                label = int(enroll_spk == eval_spk)
                if label == 0:
                    if random.randint(1,35) % 35 == 0:
                        cnn += 1
                        print(cnn)
                        #if cnn > cnt * 10: break
                        w.write(str(label)+' '+enroll_utt.split('/')[-1]+' '+eval_utt.split('/')[-1]+'\n')
print('finished!')