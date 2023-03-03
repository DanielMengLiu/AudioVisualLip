import csv
from genericpath import exists
import os
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys
import numpy as np
import shutil
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue
from moviepy.editor import *

SAMPLE_RATE = 16000
MANIFEST_DIR = "/data/liumeng/SyncLip2/data/manifest/{}_manifest.csv"
os.makedirs(os.path.dirname(MANIFEST_DIR), exist_ok = True)

def read_manifest(dataset, start = 0):
    n_speakers = 0
    rows = []
    with open(MANIFEST_DIR.format(dataset), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            rows.append([int(sid) + start, aid, filename, duration, samplerate])
            n_speakers = int(sid) + 1
    return n_speakers, rows

def save_manifest(dataset, rows):
    rows.sort()
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def create_manifest_librispeech():
    dataset = 'SLR12'
    n_speakers = 0
    log = []
    sids = dict()
    for m in ['train-clean-100', 'train-clean-360']:
        train_dataset = '/home/zeng/zeng/datasets/librispeech/{}'.format(m)
        for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
            speaker_dir = os.path.join(train_dataset, speaker)
            if os.path.isdir(speaker_dir):
                speaker = int(speaker)
                if sids.get(speaker) is None:
                    sids[speaker] = n_speakers
                    n_speakers += 1
                for task in os.listdir(speaker_dir):
                    task_dir = os.path.join(speaker_dir, task)
                    aid = 0
                    for audio in os.listdir(task_dir):
                        if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                            filename = os.path.join(task_dir, audio)
                            info = sf.info(filename)
                            log.append((sids[speaker], aid, filename, info.duration, info.samplerate))
                        aid += 1
    save_manifest(dataset, log)

def create_manifest_voxceleb1():
    dataset = 'voxceleb1'
    n_speakers = 0
    log = []
    train_dataset = '/data/datasets/voxceleb1/vox1_dev_wav'
    for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
        speaker_dir = os.path.join(train_dataset, speaker)
        aid = 0
        for sub_speaker in os.listdir(speaker_dir):
            sub_speaker_path = os.path.join(speaker_dir, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                        aid += 1
        n_speakers += 1
    save_manifest(dataset, log)

def create_manifest_voxceleb2():
    dataset = 'voxceleb2'
    n_speakers = 0
    log = []
    train_dataset = '/data/datasets/Voxceleb2/audio/dev/aac'
    for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
        speaker_dir = os.path.join(train_dataset, speaker)
        aid = 0
        for sub_speaker in os.listdir(speaker_dir):
            sub_speaker_path = os.path.join(speaker_dir, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                        aid += 1
        n_speakers += 1
    save_manifest(dataset, log)

def create_manifest_lrs3(datasets):
    src_root = '/CDShare/LRS3'
    tar_root = '/datasets1/LRS3' #'/local03/datasets/LRS3'
    for subset in datasets:
        n_speakers = 0
        log = []
        path_dataset = os.path.join(src_root, subset)
        for speaker in tqdm(os.listdir(path_dataset), desc = subset):
            speaker_dir = os.path.join(path_dataset, speaker)
            os.makedirs(os.path.join(tar_root, 'audio', subset, speaker), exist_ok=True)
            os.makedirs(os.path.join(tar_root, 'video', subset, speaker), exist_ok=True)
            vid = 0
            for video in os.listdir(speaker_dir):
                if video.endswith('.mp4'):
                    filename = os.path.join(subset, speaker, video.split('.')[0])
                    src_videopath = os.path.join(src_root, filename+'.mp4')
                    tar_videopath = os.path.join(tar_root, 'video', filename+'.mp4')
                    tar_audiopath = os.path.join(tar_root, 'audio', filename+'.wav')
                    try:
                        if not exists(tar_audiopath) or not exists(tar_videopath):
                            ## extract audio and write
                            video = VideoFileClip(src_videopath)
                            audio = video.audio
                            audio.write_audiofile(tar_audiopath)
                            shutil.copyfile(src_videopath, tar_videopath)
                    except:
                        continue
                    ## get audio information
                    info = sf.info(tar_audiopath)
                    log.append((n_speakers, vid, filename, "%.2f" % info.duration, 16000)) #info.samplerate                   
                    vid += 1
            n_speakers += 1
        save_manifest('LRS3_'+subset, log)

def select_avpair_voxceleb2():
    filepath = '/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/preprocessing/voxceleb2/liplist'
    filelist = open(filepath).read().splitlines()

    rows = {}
    new_rows = []
    with open(MANIFEST_DIR.format('voxceleb2'), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            fn = filename.replace('/datasets2/voxceleb2/audio/dev/aac/','').replace('.wav','')
            rows[fn] = [int(sid), aid, '/datasets2/voxceleb2/audio/dev/aac/'+fn+'.wav', duration, samplerate]
    for fl in tqdm(filelist):
        new_rows.append(rows[fl])

    with open('/data/liumeng/SyncLip/data/manifest/voxceleb2_5883_manifest.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

def fix_manifest():
    rows = []
    last_sid = -1
    last_aid = -1
    last_sid_ = ''
    with open('/data/liumeng/SyncLip9/data/manifest/voxceleb2mini_dev_manifest.csv', 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            if sid != last_sid_:
                last_sid = last_sid + 1
                last_aid = 0
            else:
                last_aid = last_aid + 1
            last_sid_ = sid
            rows.append([int(last_sid), int(last_aid), filename, duration, samplerate])
    with open('/data/liumeng/SyncLip9/data/manifest/voxceleb2mini_dev_manifest.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def remove_empty_file():
    rows = {}
    new_rows = []
    with open(MANIFEST_DIR.format('voxceleb2_5883'), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            filename = '/datasets3/voxceleb2/lip/dev/' + filename.replace('/datasets2/voxceleb2/audio/dev/aac/','') \
                       .replace('.wav','') + '.npz'
            if os.path.getsize(filename) <= 10000:
                os.remove(filename)
    # with open('/data/liumeng/SyncLip/data/manifest/voxceleb2_5883_manifest.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(new_rows)

def smaller_manifest():
    rows = []

    with open(MANIFEST_DIR.format('voxceleb2_perfectlandmarks'), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in tqdm(reader):
            if int(aid) < 20:
                rows.append([sid, aid, filename, duration, samplerate])
    with open(MANIFEST_DIR.format('voxceleb2_perfectlandmarks_small'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def smaller_trial_vox1test():
    videodict = []
    videos = open('/data/liumeng/SyncLip/data/manifest/voxceleb1_test_video.txt', 'r').read().splitlines()
    for vi in videos:
        if vi not in videodict:
            videodict.append(vi)
    with open('/data/liumeng/SyncLip/data/trial/veri_test2_avpair.txt', 'w') as w:
        contents = open('/data/liumeng/SyncLip/data/trial/veri_test2.txt', 'r').read().splitlines()
        for content in contents:
            label = content.split(' ')[0]
            v1 = content.split(' ')[1].replace('.wav','.mov')
            v2 = content.split(' ')[2].replace('.wav','.mov')
            if (v1 in videodict) and (v2 in videodict):
                w.write(' '.join((label, v1, v2))+'\n')
           
def vox2ChangePath_manifest():
    rows = []
    with open(MANIFEST_DIR.format('voxceleb2'), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in tqdm(reader):
            rows.append([sid, aid, filename.replace('/datasets2/voxceleb2/audio','/pcie/datasets/Voxceleb2/audio'), duration, samplerate])
    with open(MANIFEST_DIR.format('voxceleb2_106'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
                               
def create_manifest_tcdtimit():
    dataset = 'tcdtimit_fast'
    n_speakers = 0
    log = []
    train_dataset = '/home/liumeng/SSD/TCD-TIMIT/volunteers'
    for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
        speaker_dir = os.path.join(train_dataset, speaker, 'straightcam')
        aid = 0
        for audio in os.listdir(speaker_dir):
            if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                filename = os.path.join(speaker_dir, audio)
                info = sf.info(filename)
                log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                aid += 1
        n_speakers += 1
    save_manifest(dataset, log)

def create_manifest_lomgrid_dev():
    dataset = 'lomgrid_dev'
    n_speakers = -1
    log = []
    dev_list = open('/data/liumeng/xvector/data/manifest/lomgrid_devlist_audio').read().splitlines()

    last_spk = ''
    aid = 0
    for filename in tqdm(dev_list, desc = dataset):
        spk = filename.split('/')[-1].split('_')[0]
        if spk != last_spk:
            n_speakers += 1
            last_spk = spk
        info = sf.info(filename)
        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
        aid += 1
    save_manifest(dataset, log)

def create_manifest_grid():
    dataset = 'grid_test'
    audio_root = '/data/datasets/GRID/audio/'
    n_speakers = -1
    log = []
    list = open('/data/liumeng/SyncLip2/data/manifest/flist_test_grid.txt').read().splitlines()

    last_spk = ''
    aid = 0
    for filename in tqdm(list, desc = dataset):
        spk = filename.split('/')[0]
        if spk != last_spk:
            n_speakers += 1
            last_spk = spk
            aid = 0
        info = sf.info(os.path.join(audio_root, filename+'.wav'))
        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
        aid += 1
    save_manifest(dataset, log)

def create_manifest_mobilelip():
    dataset = 'mobilelip_test'
    audio_root = '/datasets2/mobilelip/audio/'
    n_speakers = -1
    log = []
    list = open('/data/liumeng/Short-Short/data/manifest/flist_test_mobilelip.txt').read().splitlines()

    last_spk = ''
    aid = 0
    for filename in tqdm(list, desc = dataset):
        spk = filename.split('/')[0]
        if spk != last_spk:
            n_speakers += 1
            last_spk = spk
            aid = 0
        info = sf.info(os.path.join(audio_root, filename+'.wav'))
        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
        aid += 1
    save_manifest(dataset, log)

def merge_manifest(datasets, dataset):
    rows = []
    n = len(datasets)
    start = 0
    for i in range(n):
        n_speakers, temp = read_manifest(datasets[i], start = start)
        rows.extend(temp)
        start += n_speakers
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def copyfile(srcpath,dstpath):
    if not os.path.isfile(srcpath):
        print("%s not exist!"%(srcpath))
    else:
        fpath,_ =os.path.split(dstpath)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        if (not os.path.exists(dstpath)) or (not os.path.getsize(dstpath)):
            shutil.copyfile(srcpath,dstpath)
            print(dstpath)
            
def create_manifest_voxceleb1_2(srcfile, dstfile):
    srclist = open(MANIFEST_DIR.format(srcfile)).read().splitlines()
    dstlist = []
    for i in tqdm(range(len(srclist))):
        src = srclist[i]
        srcspk = int(src.split(',')[0])
        srcaid = src.split(',')[1]
        srcpath = src.split(',')[2]
        srcdur = src.split(',')[3]
        srcsr = src.split(',')[4]
        if 'TCD-TIMIT' not in srcpath:
            dstspk = str(srcspk - 62)
            dst = dstspk + ',' + srcaid + ',' + srcpath + ',' + srcdur + ',' + srcsr
            dstlist.append(dst)
    with open(MANIFEST_DIR.format(dstfile), 'w') as w:
        for dl in dstlist:
            w.write(dl+'\n')
    w.close()        

def lrs3_audio_resample(subset):
    tar_root = '/datasets1/LRS3' #'/local03/datasets/LRS3'
    count = 0
    with open(MANIFEST_DIR.format('LRS3_'+subset), 'r') as f:
        reader = csv.reader(f)
        for _, _, filename, _, _ in reader:
            y, sr = sf.read(os.path.join(tar_root, 'audio', filename+'.wav'))
            if sr != 16000:
                y_16k = librosa.resample(y[:,0].astype(np.float32), sr, 16000)
                sf.write(os.path.join(tar_root, 'audio', filename+'.wav'), y_16k, 16000)
            count += 1
            print(count)
    print('done.')

def clean_manifest_voxceleb():
    filepath = '/data/liumeng/ASV-SOTA/data/manifest/voxceleb2_manifest.csv'
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            fn = filename.replace('/data/datasets/Voxceleb2/audio/dev/aac/','').replace('.wav','')
            rows.append((int(sid), aid, fn, duration, samplerate))

    with open('/data/liumeng/SyncLip3/data/manifest/voxceleb2_dev_manifest.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def vox2mini_lip():
    failure  = '/data/liumeng/SyncLip9/preprocess/voxceleb2/vox2_lip.failure'
    voxmini  = '/data/liumeng/SyncLip9/data/manifest/train_mini.txt'
    filepath = '/data/liumeng/SyncLip9/data/manifest/voxceleb2_dev_manifest.csv'
    fail_list = []
    mini_list = []
    rows = []
    
    mini = open(voxmini).read().splitlines()
    for i in range(len(mini)):
        filename = mini[i].split(' ')[1].replace('.wav','')
        mini_list.append(filename)
        
    fail = open(failure).read().splitlines()
    for i in range(len(fail)):
        filename = fail[i].replace('.npz','').replace('/datasets3/voxceleb2/landmark/dev/','')
        fail_list.append(filename)      
          
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in tqdm(reader):
            fn = filename.replace('/data/datasets/Voxceleb2/audio/dev/aac/','').replace('.wav','')
            if fn in mini_list and fn not in fail_list:
                rows.append((int(sid), aid, fn, duration, samplerate))

    with open('/data/liumeng/SyncLip9/data/manifest/voxceleb2mini_dev_manifest.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
                
if __name__ == '__main__':
    #create_manifest_lomgrid_dev()
    #create_manifest_grid()
    #create_manifest_voxceleb1()
    #create_manifest_voxceleb2()
    #merge_manifest(['tcdtimit','voxceleb1','voxceleb2'],'tcd_vox1_vox2')
    #create_manifest_voxceleb1_2('tcd_vox1_vox2_fast','vox1_vox2_fast')
    #create_manifest_tcdtimit()
    #merge_manifest(['tcdtimit_fast','vox1_vox2_fast'],'tcd_vox1_vox2_fast')
    #select_avpair_voxceleb2()
    fix_manifest()
    #smaller_manifest()
    #clean_manifest_voxceleb()
    #vox2ChangePath_manifest()
    #create_manifest_lrs3(['test'])
    # lrs3_audio_resample('test')
    #vox2mini_lip()