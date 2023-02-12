from random import randint
import numpy as np
import cv2
import os
import glob as glob
import numpy as np
import csv as csv
import tqdm as tqdm

# a = np.load('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/landmarks/lombardgrid_landmarks/s2/s2_l_bbim3a_1.npz')
# b = a['data']
# for i in range(0, 29):
#     cv2.imwrite('/data/liumeng/DeepLips/s2_l_bbim3a_1_'+str(i)+'.png', b[i])

# import torch

# target = (torch.rand(8)*2).clamp(0,1)
# print(target)
# def _load_video(filename, startframe, stopframe):
#     try:
#         if filename.endswith('npz'):
#             return np.load(filename)['data'][startframe:stopframe]
#         else:
#             return np.load(filename)
#     except IOError:
#         print( "Error when reading file: {}".format(filename) )
      
z =  np.load('/datasets1/LRS3/lip/trainval/0af00UcTOSc/50001.npz',allow_pickle=True)['data']            
for i in range(0, len(z)):
    cv2.imwrite('/data/liumeng/Short-Short/pic/p_'+str(i)+'.png', z[i])
# z_ = np.zeros((75, 96, 96))
# z_[:len(z),:,:] = z
# z = z_
# a = 1
# print(x.shape)

#print(np.random.rand(3,4,5)) 
# count = 0
# with open('/data/liumeng/SyncLip/data/manifest/voxceleb2_perfectlandmarks_manifest.csv', 'r') as f:
#     reader = csv.reader(f)
#     for sid, aid, filename, duration, samplerate in reader:
#         if randint(0,100)% 99 == 0: 
#             filename = filename.replace('/data/datasets/Voxceleb2/audio','/datasets3/voxceleb2/lip').replace('.wav','.npz').replace('/aac','')
#             x =  np.load(filename,allow_pickle=True)['data']            
#             for i in range(0, len(x)):
#                 cv2.imwrite('/datasets2/lip/'+str(count)+'_'+str(i)+'.png', x[i])     
#                 count += 1 
# os.path.getsize('/datasets2/lomgrid/lip/s12_p_lgat6s.npz')
# x = np.load('/datasets2/lomgrid/lip/s36_p_bgib1n.npz',allow_pickle=True)['data']     
# print(len(x))