# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from tqdm import tqdm
import logging
from utils.utils import read_table, read_lists
from utils.tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

def get_mean_std(emb, cohort, top_n):
    emb = emb / np.sqrt(np.sum(emb**2, axis=1, keepdims=True))
    cohort = cohort / np.sqrt(np.sum(cohort**2, axis=1, keepdims=True))
    emb_cohort_score = np.matmul(emb, cohort.T)
    emb_cohort_score = np.sort(emb_cohort_score, axis=1)[:, ::-1]
    emb_cohort_score_topn = emb_cohort_score[:, :top_n]

    emb_mean = np.mean(emb_cohort_score_topn, axis=1)
    emb_std = np.std(emb_cohort_score_topn, axis=1)

    return emb_mean, emb_std


def split_embedding(modeldir, utt_list, mean_vec):
    embs = []
    utt2idx = {}
    utt2emb = {}
    for utt in utt_list:
        emb = np.load(os.path.join(modeldir, utt.split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0].reshape(-1)
        emb = emb - mean_vec
        utt2emb[utt] = emb

    for utt in utt_list:
        embs.append(utt2emb[utt])
        utt2idx[utt] = len(embs) - 1

    return np.array(embs), utt2idx


def main(score_norm_method,
         top_n,
         modeldir,
         trial_score_file,
         score_norm_file,
         cohort_emb_manifest,
         mean_vec_path=None,
         testmode='val'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # get embedding
    if not mean_vec_path:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec = 0.0
    else:
        mean_vec_path = os.path.join(modeldir, mean_vec_path)
        assert os.path.exists(
            mean_vec_path), "mean_vec file ({}) does not exist !!!".format(
                mean_vec_path)
        mean_vec = np.load(mean_vec_path)

    # get embedding
    logging.info('get embedding ...')

    if testmode == 'val':
        enroll_list, test_list, _, _ = zip(*read_table(trial_score_file))
    elif testmode == 'test':
        enroll_list, test_list, _ = zip(*read_table(trial_score_file))
    enroll_list = sorted(list(set(enroll_list)))  # remove overlap and sort
    test_list = sorted(list(set(test_list)))
    cohort_list = sorted(list(set(read_lists(cohort_emb_manifest))))

    enroll_emb, enroll_utt2idx = split_embedding(os.path.join(modeldir,'test_emb'), enroll_list, mean_vec)
    test_emb, test_utt2idx = split_embedding(os.path.join(modeldir,'test_emb'), test_list, mean_vec)
    cohort_emb, _ = split_embedding(os.path.join(modeldir,'cohort_emb'), cohort_list, mean_vec)

    logging.info("computing normed score ...")
    if score_norm_method == "asnorm":
        top_n = top_n
    elif score_norm_method == "snorm":
        top_n = cohort_emb.shape[0]
    else:
        raise ValueError(score_norm_method)
    enroll_mean, enroll_std = get_mean_std(enroll_emb, cohort_emb, top_n)
    test_mean, test_std = get_mean_std(test_emb, cohort_emb, top_n)

    lines, normed_score_lst, final_score_lst, labels_lst = [], [], [], []
    # score norm
    with open(trial_score_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            line = line.strip().split()
            enroll_idx = enroll_utt2idx[line[0]]
            test_idx = test_utt2idx[line[1]]
            score = float(line[2])

            normed_score = 0.5 * (
                (score - enroll_mean[enroll_idx]) / enroll_std[enroll_idx]
                + (score - test_mean[test_idx]) / test_std[test_idx])
            normed_score_lst.append(normed_score)
            if testmode == 'val':
                labels_lst.append(int(line[3]))

    with open(score_norm_file, 'w', encoding='utf-8') as fout:
        min_score = min(normed_score_lst)
        max_score = max(normed_score_lst)
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip().split()
            enroll_idx = enroll_utt2idx[line[0]]
            test_idx = test_utt2idx[line[1]]

            x = normed_score_lst[i]
            normed_score = (x -min_score) / (max_score - min_score)
            final_score_lst.append(normed_score)
            fout.write('{:.5f} {} {}\n'.format(normed_score, line[0], line[1]))
    logging.info("Over!")

    if testmode == 'val':
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(final_score_lst, labels_lst, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(final_score_lst, labels_lst)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("EER: {:.6f}%, minDCF: {:.6f}".format(EER, minDCF))

if __name__ == "__main__":
    main(score_norm_method='asnorm',
         top_n=500,
         modeldir='exp/co-learning_ECAPA-TDNN_MCNN_Fbank80_ASP_emb192_2.0s/',
         trial_score_file='score/co-learning_ECAPA-TDNN_MCNN_Fbank80_ASP_emb192_2.0s/score_vox1_O.txt',
         score_norm_file='score/co-learning_ECAPA-TDNN_MCNN_Fbank80_ASP_emb192_2.0s/score_norm_vox1_O.txt',
         cohort_emb_manifest='data/manifest/cohort_manifest.csv',
         mean_vec_path=None, # None | 'submean_emb/submean.npy'
         testmode='val') # val | test  -- val has label answer, otherwise no