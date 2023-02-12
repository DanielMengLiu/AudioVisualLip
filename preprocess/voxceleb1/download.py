from typing import Dict, List, Union
from glob import glob
import youtube_dl
import os
#===== Meng: multiprocessing and logging =======
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

#============== config ==============================
root=r'/data/liumeng/SyncLip/preprocess/voxceleb1/txt'    ## your txt directory path 
root1=r'/data/liumeng/SyncLip/preprocess/voxceleb1/video' ## your saved video path
num_workers = 10

def read_file_name(file_dir): # 读取每个文件名
    name=os.listdir(file_dir)
    return name

def find_txt(directory,pattern='*.txt'): # 读取txt文件
    """Recursively finds all waves matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def timeCost(seconds): # 将帧差转化为标准秒
    m,s=divmod(seconds,60)
    h,m=divmod(m,60)
    return h,m,s

def run(iStart, iEnd, id, v_list, queue=0):
    pathlist=[]
    urllist=[]
    for speaker in id[iStart:iEnd]:
        path=root+'/'+speaker
        pathlist.append(path)
        name=read_file_name(path)
        for i in name :
            url='https://www.youtube.com/watch?v='+i
            urlpath=path+'/'+i
            urllist.append(url)
            txt=read_file_name(urlpath)
            for file in txt :
                try:
                    filepath=urlpath+'/'+file
                    storepath=root1+'/'+speaker+'/'+i+'/'+file.strip('.txt')
                    storename=storepath+'.mp4'
                    storename2=storepath+'.mov'
                    if os.path.exists(storename) or os.path.exists(storename2):
                        continue
                    f=open(filepath,'r')
                    line=f.readlines()
                    start=line[7].split(" ")
                    end=line[-1].split(" ")
                    start=round((int(start[0]))/25)
                    end=round((int(end[0]))/25)
                    h1,m1,s1=timeCost(start)
                    h2,m2,s2=timeCost(end)
                    h1=str(h1)
                    m1=str(m1)
                    s1=str(s1)
                    h2=str(h2)
                    m2=str(m2)
                    s2=str(s2)
                    ydl_opts = {
                        'outtmpl':storename,
                        'format': 'worst',
                        'postprocessors': [{
                            'key': 'FFmpegVideoConvertor',
                            'preferedformat': 'mov',
                        }],
                        'postprocessor_args': [
                            "-ss", h1+':'+m1+':'+s1, "-to", h2+':'+m2+':'+s2  # 最后为文件名
                        ],
                    }
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                except Exception as e:
                    logging.warning(e)
                    v_list.append(storepath)
            
            
if __name__ == '__main__':
    abnorm_list = list()
    id=read_file_name(root) #读取说话人id文件名
    
    v_list = Manager().list()
    queues = [Queue(ctx=multiprocessing.get_context()) for i in range(num_workers)]

    part = list(range(0, len(id)+1, int(len(id)//num_workers)))
    part[-1] = len(id)
    
    args = [(part[i], part[i+1], id, v_list, queues[i]) for i in range(num_workers)]

    jobs = [Process(target=run, args=(a)) for a in args]
    for j in jobs: j.start()
    for j in jobs: j.join()        
    
    for fn in v_list:
        abnorm_list.append(fn)
    with open('voxceleb1_test_downloadfailure.log','w+') as w:
        for fn in abnorm_list:
            w.writelines(fn+'\n')

