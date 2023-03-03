import os
import csv
import glob
import numpy as np
import random, math
import soundfile as sf
import torch, torchaudio
from torch.utils.data import Dataset
from models.preprocess_new import *
from models.preprocess_new import FeatureAug
from models.feature_new import OnlineFbank
from utils.utils import read_video_gray


class AudioTrainset(Dataset):
    def __init__(self, audioopts):
        self.second_range = audioopts['seconds']
        self.num_second = random.randint(self.second_range[0], self.second_range[1]) 
        TRAIN_MANIFEST = audioopts['train_manifest']

        #audio config
        self.sample_rate = audioopts['sample_rate']
        self.audiodata_dir = audioopts['train_audiodir']
        self.audiodata_suffix = '.wav'
        self.audioopts = audioopts
        
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(TRAIN_MANIFEST).read().splitlines()
        dictkeys = list(set([x.split(',')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split(',')[0]]
            file_name     = line.split(',')[2]
            # math trick:num_segments=int(audio_len//num_second + 1)
            num_segments  = round(float(line.split(',')[3])/self.num_second + 0.5)
            for i in range(num_segments):
                self.data_label.append(speaker_label)
                self.data_list.append(file_name + '___' + str(i))  # 
        self.n_spk = len(set(self.data_label))
        
        if 'noiseaug' in audioopts.keys():
            # Load and configure augmentation files
            musan_path = audioopts['musan_path']
            rir_path = audioopts['rir_path']
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
            self.noiselist = {}
            augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            for file in augment_files:
                if file.split('/')[-4] not in self.noiselist:
                    self.noiselist[file.split('/')[-4]] = []
                self.noiselist[file.split('/')[-4]].append(file)
            self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        self.noiseaug_prob = audioopts['noiseaug'] if 'noiseaug' in audioopts.keys() else 0.0
        self.speeds = audioopts['speedperturb'] if 'speedperturb' in audioopts.keys() else [1.0]
        self.feature = OnlineFbank()
        self.feataug = FeatureAug() # Spec augmentation
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        audio, sr = sf.read(os.path.join(self.audiodata_dir, self.data_list[index].split('___')[0] + self.audiodata_suffix)) #
        length_audio = self.num_second * 16000 + 240
        if audio.shape[0] <= length_audio:
            shortage = length_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_audiopoint = np.int64(random.random()*(audio.shape[0]-length_audio))
        audio = audio[start_audiopoint:start_audiopoint + length_audio]
        audiofeat = np.stack([audio],axis=0)
        
        if 'noiseaug' in self.audioopts.keys():            
            augtype = random.randint(0,2)
            if augtype == 0:   # Original clean speech
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 1: # Noise and reverb augmentation
                noiseaugtype = random.randint(0,4)
                if noiseaugtype == 0: # Noise
                    audiofeat = add_noise(audiofeat, 'noise', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 1: # Music
                    audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 2: # speech
                    audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 3: # Television noise
                    audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                    audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 4: # Reverberation
                    audiofeat = add_rev(audiofeat, self.num_second, self.rir_files)
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 2: # spec augmentation
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
                audiofeat = self.feataug(audiofeat)
        elif 'specaug' in self.audioopts.keys():            
            augtype = random.randint(0,1)
            if augtype == 0:   # Original clean speech
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 1: # spec augmentation
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
                audiofeat = self.feataug(audiofeat)
        else:
            audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
            audiofeat = self.feature(audiofeat)        
        return audiofeat.squeeze(0), self.data_label[index]

class VisualLipTrainset(Dataset):
    def __init__(self, videoopts):
        self.second_range = videoopts['seconds']
        self.num_second = random.randint(self.second_range[0], self.second_range[1]) 
        TRAIN_MANIFEST = videoopts['train_manifest']

        self.videodata_dir = videoopts['train_videodir']
        self.videodata_suffix = '.mp4'
        self.video_fps = 25
        
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(TRAIN_MANIFEST).read().splitlines()
        dictkeys = list(set([x.split(',')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split(',')[0]]
            file_name     = line.split(',')[2]
            num_segments  = round(float(line.split(',')[3])/self.num_second + 0.5) # math trick
            for i in range(num_segments):
                self.data_label.append(speaker_label)
                self.data_list.append(file_name + '___' + str(i)) #
        self.n_spk = len(set(self.data_label))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        videofeats = []
        # videofilename = os.path.join(self.videodata_dir, self.data_list[index].split('.')[0].split('___')[0]) + self.videodata_suffix #
        videofilename = os.path.join(self.videodata_dir, self.data_list[index].split('___')[0]) + self.videodata_suffix #

        if os.path.exists(videofilename):
            # video = np.load(videofilename, allow_pickle=True)['data']
            video = read_video_gray(videofilename)
        else:
            video = np.random.rand(50,96,96)
        length_video = math.floor(self.num_second * self.video_fps)
        if video.shape[0] < length_video:
            shortage = length_video - video.shape[0]
            video = np.pad(video, ((0, shortage), (0, 0), (0, 0)), 'wrap')

        start_videoframe = np.int64(random.random()*(video.shape[0]-length_video))
        video = video[start_videoframe:start_videoframe + length_video]
        # augmentation
        video = get_preprocessing_pipelines()['train'](video)  
        videofeat = np.stack(video,axis=0)
        videofeats.append(videofeat)
        videofeats = np.array(videofeats).astype(np.float32)

        return torch.from_numpy(videofeats), self.data_label[index]

class AudioVisualLipTrainset(Dataset):
    def __init__(self, audioopts):
        self.second_range = audioopts['seconds']
        self.num_second = random.randint(self.second_range[0], self.second_range[1]) 

        TRAIN_MANIFEST = audioopts['train_manifest']
        self.audioopts =audioopts
        #audio config
        self.sample_rate = audioopts['sample_rate']
        self.audiodata_dir = audioopts['train_audiodir']
        self.videodata_dir = audioopts['train_videodir']
        self.audiodata_suffix = '.wav'
        self.videodata_suffix = '.mp4'
        self.video_fps = 25
        
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(TRAIN_MANIFEST).read().splitlines()
        dictkeys = list(set([x.split(',')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split(',')[0]]
            file_name     = line.split(',')[2]
            num_segments  = round(float(line.split(',')[3])/self.num_second + 0.5) # math trick
            for i in range(num_segments):
                self.data_label.append(speaker_label)
                self.data_list.append(file_name + '___' + str(i)) # 
        self.n_spk = len(set(self.data_label))
        
        if 'noiseaug' in audioopts.keys():
            # Load and configure augmentation files
            musan_path = audioopts['musan_path']
            rir_path = audioopts['rir_path']
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
            self.noiselist = {}
            augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            for file in augment_files:
                if file.split('/')[-4] not in self.noiselist:
                    self.noiselist[file.split('/')[-4]] = []
                self.noiselist[file.split('/')[-4]].append(file)
            self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        self.noiseaug_prob = audioopts['noiseaug'] if 'noiseaug' in audioopts.keys() else 0.0
        self.speeds = audioopts['speedperturb'] if 'speedperturb' in audioopts.keys() else [1.0]
        self.feature = OnlineFbank()
        self.feataug = FeatureAug() # Spec augmentation
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        # random select a frame number in uniform distribution
        audiofilename = os.path.join(self.audiodata_dir, self.data_list[index].split('___')[0]) + self.audiodata_suffix #
        videofilename = os.path.join(self.videodata_dir, self.data_list[index].split('___')[0]) + self.videodata_suffix #

        # audio sampling
        audio, sr = sf.read(audiofilename)	
        if len(self.speeds) > 1:
            print('speaker augmentation is not allowed in audio-visual training')
        length_audio = self.num_second * 16000 + 240
        if audio.shape[0] <= length_audio:
            shortage = length_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_audiopoint = np.int64(random.random()*(audio.shape[0]-length_audio))
        audio = audio[start_audiopoint:start_audiopoint + length_audio]
        audiofeat = np.stack([audio],axis=0)

        if 'noiseaug' in self.audioopts.keys():            
            augtype = random.randint(0,2)
            if augtype == 0:   # Original clean speech
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 1: # Noise and reverb augmentation
                noiseaugtype = random.randint(0,4)
                if noiseaugtype == 0: # Noise
                    audiofeat = add_noise(audiofeat, 'noise', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 1: # Music
                    audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 2: # speech
                    audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 3: # Television noise
                    audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                    audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
                elif noiseaugtype == 4: # Reverberation
                    audiofeat = add_rev(audiofeat, self.num_second, self.rir_files)
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 2: # spec augmentation
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
                audiofeat = self.feataug(audiofeat)
        elif 'specaug' in self.audioopts.keys():            
            augtype = random.randint(0,1)
            if augtype == 0:   # Original clean speech
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
            elif augtype == 1: # spec augmentation
                audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
                audiofeat = self.feature(audiofeat)
                audiofeat = self.feataug(audiofeat)
        else:
            audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
            audiofeat = self.feature(audiofeat)
            
        if os.path.exists(videofilename):
            # video = np.load(videofilename, allow_pickle=True)['data']
            video = read_video_gray(videofilename)
        else:
            video = np.random.rand(50,96,96)
        length_video = math.floor(self.num_second * self.video_fps)
        if video.shape[0] < length_video:
            shortage = length_video - video.shape[0]
            video = np.pad(video, ((0, shortage), (0, 0), (0, 0)), 'wrap')
        
        start_videoframe = math.floor((start_audiopoint / sr) * self.video_fps) # corresponds to audio sampling
        if start_videoframe + length_video < len(video):
            video = video[start_videoframe:start_videoframe + length_video]
        else:
            video = video[-length_video:]
            
        video = get_preprocessing_pipelines()['train'](video)  
        videofeat = np.stack(video,axis=0)
        videofeat = np.array(videofeat).astype(np.float32)
        
        return audiofeat.squeeze(0), torch.from_numpy(videofeat).unsqueeze(0), self.data_label[index]

# class CrossModalLipTrainset(Dataset):
#     def __init__(self, audioopts):
#         self.second_range = audioopts['seconds']
#         self.num_second = random.randint(self.second_range[0], self.second_range[1]) 

#         TRAIN_MANIFEST = audioopts['train_manifest']
#         self.audioopts =audioopts
#         #audio config
#         self.sample_rate = audioopts['sample_rate']
#         self.audiodata_dir = audioopts['train_audiodir']
#         self.videodata_dir = audioopts['train_videodir']
#         self.audiodata_suffix = '.wav'
#         self.videodata_suffix = '.npz'
#         self.video_fps = 25
        
#         # Load data & labels
#         self.data_list  = {}
#         self.data_label = []
#         self.count = 0
#         lines = open(TRAIN_MANIFEST).read().splitlines()
#         dictkeys = list(set([x.split(',')[0] for x in lines]))
#         dictkeys.sort()
#         dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
#         for index, line in enumerate(lines):
#             speaker_label = dictkeys[line.split(',')[0]]
#             file_name     = line.split(',')[2]
#             num_segments  = round(float(line.split(',')[3])/self.num_second + 0.5) # math trick
#             if speaker_label not in self.data_list.keys():
#                 self.data_list[speaker_label] = []
#             for i in range(num_segments):
#                 self.data_label.append(speaker_label)
#                 self.data_list[speaker_label].append(file_name + '___' + str(i))
#                 self.count += 1
#         self.data_label_unique = list(set(self.data_label))
#         self.n_spk = len(set(self.data_label))
        
#         if 'noiseaug' in audioopts.keys():
#             # Load and configure augmentation files
#             musan_path = audioopts['musan_path']
#             rir_path = audioopts['rir_path']
#             self.noisetypes = ['noise','speech','music']
#             self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
#             self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
#             self.noiselist = {}
#             augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
#             for file in augment_files:
#                 if file.split('/')[-4] not in self.noiselist:
#                     self.noiselist[file.split('/')[-4]] = []
#                 self.noiselist[file.split('/')[-4]].append(file)
#             self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
#         self.noiseaug_prob = audioopts['noiseaug'] if 'noiseaug' in audioopts.keys() else 0.0
#         self.speeds = audioopts['speedperturb'] if 'speedperturb' in audioopts.keys() else [1.0]
#         self.feature = OnlineFbank()
#         self.feataug = FeatureAug() # Spec augmentation
            
#     def __len__(self):
#         if self.count < 1090000: # lrs3
#             return self.count
#         else:                             # vox2
#             return 1090000 #len(self.data_list)

#     def __collate_fn__(self, batch):
#         #random.seed(index)
#         # prop = random.randint(0,2)
#         # adata, vdata = '', '' # init
#         # if prop == 0: ## positive pair - label is 1
#         #     spkindex = random.choice(self.data_label_unique)
#         #     aspkindex, vspkindex = spkindex, spkindex
#         #     while(1):
#         #         datapair = random.choices(self.data_list[spkindex], k=2)
#         #         adata, vdata = datapair[0], datapair[1]
#         #         if (adata != vdata) or len(self.data_list[spkindex])==1: break
#         # else:         ## negative pair - label is 0
#         #     aspkindex, vspkindex = 0, 0
#         #     while(1):
#         #         spkpair = random.choices(self.data_label_unique, k=2)
#         #         aspkindex, vspkindex = spkpair[0], spkpair[1]
#         #         if aspkindex != vspkindex: break
#         #     adata = random.choice(self.data_list[aspkindex])
#         #     vdata = random.choice(self.data_list[vspkindex])
#         for sid in batch:
#             spkindex = sid #random.choice(self.data_label_unique)
#             aspkindex, vspkindex = spkindex, spkindex
#             datapair = random.choice(self.data_list[spkindex])
#             adata, vdata = datapair, datapair
#             # random select a frame number in uniform distribution
#             audiofilename = os.path.join(self.audiodata_dir, adata.split('.')[0].split('___')[0]) + self.audiodata_suffix #


#             # audio sampling
#             audio, sr = sf.read(audiofilename)	
#             if len(self.speeds) > 1:
#                 print('speaker augmentation is not allowed in audio-visual training')
#             length_audio = self.num_second * 16000 + 240
#             if audio.shape[0] <= length_audio:
#                 shortage = length_audio - audio.shape[0]
#                 audio = np.pad(audio, (0, shortage), 'wrap')
#             start_audiopoint = np.int64(random.random()*(audio.shape[0]-length_audio))
#             audio = audio[start_audiopoint:start_audiopoint + length_audio]
#             audiofeat = np.stack([audio],axis=0)

#             if 'noiseaug' in self.audioopts.keys():            
#                 augtype = random.randint(0,2)
#                 if augtype == 0:   # Original clean speech
#                     audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                     audiofeat = self.feature(audiofeat)
#                 elif augtype == 1: # Noise and reverb augmentation
#                     noiseaugtype = random.randint(0,4)
#                     if noiseaugtype == 0: # Noise
#                         audiofeat = add_noise(audiofeat, 'noise', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
#                     elif noiseaugtype == 1: # Music
#                         audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
#                     elif noiseaugtype == 2: # speech
#                         audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
#                     elif noiseaugtype == 3: # Television noise
#                         audiofeat = add_noise(audiofeat, 'speech', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
#                         audiofeat = add_noise(audiofeat, 'music', self.numnoise, self.noiselist, self.noisesnr, self.num_second)
#                     elif noiseaugtype == 4: # Reverberation
#                         audiofeat = add_rev(audiofeat, self.num_second, self.rir_files)
#                     audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                     audiofeat = self.feature(audiofeat)
#                 elif augtype == 2: # spec augmentation
#                     audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                     audiofeat = self.feature(audiofeat)
#                     audiofeat = self.feataug(audiofeat)
#             elif 'specaug' in self.audioopts.keys():            
#                 augtype = random.randint(0,1)
#                 if augtype == 0:   # Original clean speech
#                     audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                     audiofeat = self.feature(audiofeat)
#                 elif augtype == 1: # spec augmentation
#                     audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                     audiofeat = self.feature(audiofeat)
#                     audiofeat = self.feataug(audiofeat)
#             else:
#                 audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0)
#                 audiofeat = self.feature(audiofeat)
                
#             if os.path.exists(videofilename):
#                 video = np.load(videofilename, allow_pickle=True)['data']
#             else:
#                 video = np.random.rand(50,96,96)
#             length_video = math.floor(self.num_second * self.video_fps)
#             if video.shape[0] < length_video:
#                 shortage = length_video - video.shape[0]
#                 video = np.pad(video, ((0, shortage), (0, 0), (0, 0)), 'wrap')
            
#             start_videoframe = math.floor((start_audiopoint / sr) * self.video_fps) # corresponds to audio sampling
#             if start_videoframe + length_video < len(video):
#                 video = video[start_videoframe:start_videoframe + length_video]
#             else:
#                 video = video[-length_video:]
                
#             video = get_preprocessing_pipelines()['train'](video)  
#             videofeat = np.stack(video,axis=0)
#             videofeat = np.array(videofeat).astype(np.float32)
        
#         return audiofeat.squeeze(0), torch.from_numpy(videofeat).unsqueeze(0), aspkindex, vspkindex        
         
#     def __getitem__(self, index):
#         return index % self.n_spk  # different spks per batch

class AudioTestset(Dataset):
    def __init__(self, audioopts, utts, stage):
        self.path = audioopts['test_trial']

        if stage == 'cohort' or stage == 'submean':
            self.audiodata_dir = audioopts['train_audiodir']
        elif stage == 'val' or stage == 'test':
            self.audiodata_dir = audioopts['test_audiodir']
        self.audiodata_suffix = '.wav'
        self.utts = utts
        # for GRID datasets
        self.resample = torchaudio.transforms.Resample(44100, 16000)
        self.feature = OnlineFbank()
        
    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]

        audioutt_path = os.path.join(self.audiodata_dir, utt+self.audiodata_suffix)
        #audio sampling
        audio, sr = sf.read(audioutt_path)
        if sr != 16000:
            audio = torch.from_numpy(audio[:,1].reshape(-1).astype(np.float32))
            audio = self.resample(audio)            
        # global utterance
        data_1 = torch.FloatTensor(np.stack([audio],axis=0)).unsqueeze(1)
        data_1 = self.feature(data_1)
        
        # local utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis=0).astype(np.float)
        data_2 = torch.FloatTensor(feats).unsqueeze(1)
        data_2 = self.feature(data_2)
        
        return data_1, data_2, utt

class VisualLipTestset(Dataset):
    def __init__(self, videoopts, utts, stage):
        self.path = videoopts['test_trial']

        if stage == 'val' or stage == 'test':
            self.videodata_dir = videoopts['test_videodir']
        self.videodata_suffix = '.mp4'
        self.utts = utts

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        videoutt_path = os.path.join(self.videodata_dir, utt+self.videodata_suffix)
        #video sampling
        if os.path.exists(videoutt_path):
            # video = np.load(videoutt_path)['data']
            video = read_video_gray(videoutt_path)
        else:
            print('no lip') #video = np.random.rand(50,96,96)
        # global utterance
        data_1 = torch.FloatTensor(video).unsqueeze(0)

        # local utterance matrix
        max_video = 50
        if video.shape[0] < max_video:
            shortage = max_video - video.shape[0]
            video = np.pad(video, ((0, shortage), (0, 0), (0, 0)), 'wrap')
        feats = []
        startframe = np.linspace(0, video.shape[0]-max_video, num=5)
        for asf in startframe:
            feats.append(video[int(asf):int(asf)+max_video])
        feats = np.stack(feats, axis=0).astype(np.float)
        data_2 = torch.FloatTensor(feats).unsqueeze(1)

        return data_1, data_2, utt

class AudioVisualLipTestset(Dataset):
    def __init__(self, opts, utts, stage):
        self.path = opts['test_trial']

        if stage == 'val' or stage == 'test':
            self.audiodata_dir = opts['test_audiodir']
            self.videodata_dir = opts['test_videodir']
        self.audiodata_suffix = '.wav'
        self.videodata_suffix = '.mp4'
        self.utts = utts
        # for GRID datasets
        self.resample = torchaudio.transforms.Resample(44100, 16000)
        self.feature = OnlineFbank()
        
    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        
        audioutt_path = os.path.join(self.audiodata_dir, utt+self.audiodata_suffix)
        #audio sampling
        audio, sr = sf.read(audioutt_path)
        if sr != 16000:
            audio = torch.from_numpy(audio[:,1].reshape(-1).astype(np.float32))
            audio = self.resample(audio)      
        # global utterance
        adata_1 = torch.FloatTensor(np.stack([audio],axis=0)).unsqueeze(1)
        adata_1 = self.feature(adata_1)
        
        # local utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis=0).astype(np.float)
        adata_2 = torch.FloatTensor(feats).unsqueeze(1)
        adata_2 = self.feature(adata_2)
        
        videoutt_path = os.path.join(self.videodata_dir, utt+self.videodata_suffix)
        #video sampling
        if os.path.exists(videoutt_path):
            # video = np.load(videoutt_path)['data']
            video = read_video_gray(videoutt_path)
        else:
            print('no lip') #video = np.random.rand(50,96,96)
        # global utterance
        vdata_1 = torch.FloatTensor(video).unsqueeze(0)

        # local utterance matrix
        max_video = 50
        if video.shape[0] < max_video:
            shortage = max_video - video.shape[0]
            video = np.pad(video, ((0, shortage), (0, 0), (0, 0)), 'wrap')
        feats = []
        startframe = np.linspace(0, video.shape[0]-max_video, num=5)
        for asf in startframe:
            feats.append(video[int(asf):int(asf)+max_video])
        feats = np.stack(feats, axis=0).astype(np.float)
        vdata_2 = torch.FloatTensor(feats).unsqueeze(1)

        return adata_1, adata_2, vdata_1, vdata_2, utt

class Vox2Submeanset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz
        '''
        opts_audio = opts['audio_feature']
        self.path = opts['train_manifest']
        #audio config
        self.rate = opts_audio['rate']
        self.feat_type = opts_audio['feat_type']
        self.opts_audio = opts_audio[self.feat_type] # can choose mfcc or fbank as input feat
        self.audiodata_dir = opts['audiodata_dir']
        self.audiodata_suffix = '.wav'

        self.utts = []      
        self.count = 0
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            for _, _, filename, _, _ in reader:
                self.utts.append(filename)
                self.count += 1

    def fix_length(self, feat, max_length=300):
        max_length = max_length * 160 + 240
        out_feat = feat
        while out_feat.shape[0] < max_length:
            out_feat = np.concatenate((out_feat, feat), axis=0)
        feat_len = out_feat.shape[0]
        start = random.randint(a=0, b=feat_len-max_length)
        end = start + max_length
        out_feat = out_feat[start:end,]
        return out_feat

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        audioutt_path = os.path.join(self.audiodata_dir, utt+self.audiodata_suffix)
        #audio sampling
        audiofeat, rate = sf.read(audioutt_path)
        audiofeat = self.fix_length(audiofeat, max_length=600)
        audiofeat = np.array(audiofeat).astype(np.float32)
        return torch.from_numpy(audiofeat).unsqueeze(0), utt




     