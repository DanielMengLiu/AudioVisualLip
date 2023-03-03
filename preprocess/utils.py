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
    
def read_video(filename):
    cap = cv2.VideoCapture(filename)                                             
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            yield frame                                                          
        else:                                                                    
            break                                                                
    cap.release()
