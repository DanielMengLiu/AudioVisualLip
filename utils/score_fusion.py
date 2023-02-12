import os, sys, tqdm
import numpy as np
from eval_metrics_new import *
from tools import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

def eer_cos_scorefusion(scorelist):
    length = len(scorelist)
    if length <= 1: assert 'must use 2 or more score files!'
    scores = []
    y_true = []
    best_eer = 100  # initial poor value
    best_weight = [100] * length
    for i in range(length):
        scorefile = os.path.join('score', scorelist[i])
        scores.append([])
        with open(scorefile, 'r') as f:
            for line in f:
                line = line.rstrip()
                _, _, score, label = line.split(' ')
                if i == 0:
                    y_true.append(eval(label))
                scores[i].append(eval(score))
    if length == 2:
        for w1 in np.arange(0,1,0.1):
            for w2 in np.arange(0,1,0.1):
                    y_pred = []
                    for i in range(len(y_true)):
                        score = w1 * scores[0][i] + w2 * scores[1][i]
                        y_pred.append(score)           
                    eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
                    fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
                    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)       
                    if eer <= best_eer:
                        print(eer)
                        best_eer = eer
                        best_weight[0] = w1
                        best_weight[1] = w2
        print('The best alpha beta are: %.2f and %.2f' % (best_weight[0], best_weight[1]))
        y_pred = []
        for i in range(len(y_true)):
            score = best_weight[0] * scores[0][i] + best_weight[1] * scores[1][i]
            y_pred.append(score)           
        eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)    
    if length == 3:
        for w1 in np.arange(0,1,0.1):
            for w2 in np.arange(0,1,0.1):
                for w3 in np.arange(0,1,0.1):
                    y_pred = []
                    for i in range(len(y_true)):
                        score = w1 * scores[0][i] + w2 * scores[1][i] + w3 * scores[2][i]
                        y_pred.append(score)           
                    eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
                    fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
                    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)       
                    if eer <= best_eer:
                        print(eer)
                        best_eer = eer
                        best_weight[0] = w1
                        best_weight[1] = w2
                        best_weight[2] = w3
        print('The best alpha beta and gama are: %.2f , %.2f and %.2f' % (best_weight[0], best_weight[1], best_weight[2]))
        y_pred = []
        for i in range(len(y_true)):
            score = best_weight[0] * scores[0][i] + best_weight[1] * scores[1][i] + best_weight[2] * scores[2][i]
            y_pred.append(score)           
        eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)     
    if length == 4:
        for w1 in np.arange(0.5,1,0.1): # w1 = 0.9 #
            for w2 in np.arange(0,1,0.1):
                for w3 in np.arange(0,1,0.1):
                    for w4 in np.arange(0,1,0.1):
                        y_pred = []
                        for i in range(len(y_true)):
                            score = w1 * scores[0][i] + w2 * scores[1][i] + w3 * scores[2][i] + w4 * scores[3][i]
                            y_pred.append(score)           
                        eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
                        fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
                        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)       
                        if eer <= best_eer:
                            print(eer)
                            best_eer = eer
                            best_weight[0] = w1
                            best_weight[1] = w2
                            best_weight[2] = w3
                            best_weight[3] = w4
        print('The best alpha beta and gama are: %.2f, %.2f, %.2f and %.2f' % (best_weight[0], best_weight[1], best_weight[2], best_weight[3]))
        y_pred = []
        for i in range(len(y_true)):
            score = best_weight[0] * scores[0][i] + best_weight[1] * scores[1][i] + best_weight[2] * scores[2][i]+ best_weight[3] * scores[3][i]
            # score = 0.9 * scores[0][i] + 0.3 * scores[1][i] + 0.1 * scores[2][i]+ 0.2 * scores[3][i]
            y_pred.append(score)           
        eer = tuneThresholdfromScore(y_pred, y_true, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(y_pred, y_true)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)    
                     
    with open('scorefusion.txt', 'w') as fh: 
        for (i, (k, v)) in enumerate(zip(y_true, y_pred)):
            fh.write('{} {}\n'.format(k, v))   
    print("EER: {:.6f}%, minDCF: {:.6f}".format(eer, minDCF))


# eer_cos_scorefusion(['lrs3_cross-modal_ECAPA-TDNN_MCNN_3aug3blocks/a_score_lrs3_O.txt',\
#                     'lrs3_cross-modal_ECAPA-TDNN_MCNN_3aug3blocks/v_score_lrs3_O.txt',\
#                     'lrs3_cross-modal_ECAPA-TDNN_MCNN_3aug3blocks/transa_score_lrs3_O.txt',\
#                     'lrs3_cross-modal_ECAPA-TDNN_MCNN_3aug3blocks/transv_score_lrs3_O.txt'])

# eer_cos_scorefusion(['lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_av_lrs3_O.txt',\
#                     'lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_a\'v\'_lrs3_O.txt'])

eer_cos_scorefusion(['lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_a_vox1_O-29.txt',\
                     'lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_v_vox1_O-29.txt',\
                     'lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_a\'_vox1_O-29.txt',\
                     'lrs3_cross-modal-pretrained_ECAPA-TDNN_MCNN_noisespec3blocks/score_v\'_vox1_O-29.txt'])