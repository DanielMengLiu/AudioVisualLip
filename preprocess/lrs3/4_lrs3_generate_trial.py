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
    parser.add_argument('--test-manifest', default='/data/liumeng/SyncLip2/data/manifest/LRS3_test_manifest.csv', help='test utt list')
    parser.add_argument('--trial', default='/data/liumeng/SyncLip2/data/trial/lrs3_E.trl', help='lrs3 test set trial')
    parser.add_argument('--num-pairs', default=20000, help='number of trial pairs')
    parser.add_argument('--percent-target', type=float, default=0.5, help='target and non-target 0.0~1.0')
  
    args = parser.parse_args()
    return args

args = load_args()

def find_target(fn1):
    spk1 = fn1.split('/')[-2]
    cnt = 0
    while True:
        fn2 = random.sample(utts[spk1],1)[0]
        if fn2 != fn1 and (fn1,fn2) not in memory and (fn2,fn1) not in memory:
            memory.append((fn1,fn2))
            return fn2
        cnt += 1
        if cnt > 20:
            return None
    

def find_nontarget(fn1):
    spk1 = fn1.split('/')[-2]
    cnt = 0
    while True:
        fn2 = random.sample(lutts,1)[0]
        random_spk = fn2.split('/')[-2]
        if random_spk == spk1:
            continue
        if fn2 != fn1 and (fn1,fn2) not in memory and (fn2,fn1) not in memory:
            memory.append((fn1,fn2))
            return fn2
        cnt += 1
        if cnt > 20:
            return None


utts = {}
with open(args.test_manifest, mode='r',newline='') as r:
    reader = csv.reader(r)
    for _,_,fn,_,_ in reader:
        spk = fn.split('/')[-2]
        if spk not in utts.keys():
            utts[spk] = []
        utts[spk].append(fn)
lutts = [i for i in utts.values()]
lutts = reduce(lambda x, y: x.extend(y) or x, [ i if isinstance(i, list) else [i] for i in lutts])
trial_list = np.random.choice(lutts, args.num_pairs).tolist()

num_target = int(args.percent_target * args.num_pairs)
num_nontarget = int((1.0-args.percent_target) * args.num_pairs)

memory = []
pairs = 0
with open(args.trial, 'w') as w:
    # target
    for fn1 in trial_list[0:num_target]:
        if len(utts[fn1.split('/')[-2]]) > 1:
            fn2 = find_target(fn1) 
            if fn2 is not None:
                w.write('1'+' '+fn1+' '+fn2+'\n')
                pairs += 1
                print(pairs)
    # non-target
    for fn1 in trial_list[num_target:]:
        if len(utts[fn1.split('/')[-2]]) > 1:
            fn2 = find_nontarget(fn1)
        if fn2 is not None:
            w.write('0'+' '+fn1+' '+fn2+'\n')
            pairs += 1
            print(pairs)
w.close()

print('total pairs:', pairs)