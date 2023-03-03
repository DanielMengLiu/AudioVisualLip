#coding=utf-8
import os
import cv2
import numpy as np


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content

def save2npz(filename, data=None):                                               
    assert data is not None, "data is {}".format(data)                           
    if not os.path.exists(os.path.dirname(filename)):                            
        os.makedirs(os.path.dirname(filename))                                 
    #np.savez_compressed(filename, data=data)  # error core dump
    np.savez_compressed(filename, data=data)

def save2jpg(filename, data=None):                                               
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(filename):
        os.makedirs(filename.replace(filename.split('/')[-1],''), exist_ok=True)    
    for i in range(0,len(data)):
        cv2.imwrite(filename.replace('.', '_'+str(i)+'.'), data[i])

def save2mp4(filename, data=None, convert_gray=True):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(filename):
        os.makedirs(filename.replace(filename.split('/')[-1], ''), exist_ok=True)
    data = np.array(data)  # ndarray for getting shape
    fps = 25
    codec_id = "mp4v"  # ID for a video codec.
    fourcc = cv2.VideoWriter_fourcc(*codec_id)
    if convert_gray:
        num_frames, height, width = data.shape
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height), 0)  # last arg 0 for gray video
        for i in range(num_frames):
            out.write(data[i, ...])
    else:
        num_frames, height, width, channel = data.shape
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for i in range(num_frames):
            out.write(data[i, ..., ::-1])
    out.release()

def read_video(filename):
    cap = cv2.VideoCapture(filename)                                             
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            yield frame                                                          
        else:                                                                    
            break                                                                
    cap.release()
