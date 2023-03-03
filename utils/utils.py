import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# -- Media utils
def extract_opencv(filename, bgr=False):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video if bgr else video[..., ::-1]


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content


def load_json( json_fp ):
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def read_lists(list_file):
    """ list_file: only 1 column
    """
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_table(table_file):
    """ table_file: any columns
    """
    table_list = []
    with open(table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            table_list.append(tokens)
    return table_list

def read_video_gray(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video