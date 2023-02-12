import os, sys, torch, csv
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from models.loss import AAMsoftmax

def changelabelmap(w):
    # mimic tao's Loading data & labels
    data_label_old = []
    lines = open('data/manifest/train_list.txt').read().splitlines()
    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
    for index, line in enumerate(lines):
        file_name     = os.path.join(line.split()[1])
        data_label_old.append(file_name.split('/')[0])
    data_label_old_ = list(set(data_label_old))
    data_label_old_.sort(key=data_label_old.index)
    
    data_label_new = []           
    dataset = []
    current_sid = -1
    with open('data/manifest/voxceleb2_dev_manifest.csv', 'r') as f:
        reader = csv.reader(f)
        for sid, _, filename, duration, samplerate in reader:
            if sid != current_sid:
                dataset.append([])
                current_sid = sid
                data_label_new.append(filename.split('/')[0])
    data_label_new_ = list(set(data_label_new))
    data_label_new_.sort(key=data_label_new.index)
    
    w_new = []
    for i in range(len(data_label_new_)):
        label_new = data_label_new_[i]
        index_old = data_label_old_.index(label_new)
        w_new.append(w[index_old])
    w_new = torch.stack(w_new)
    return w_new
    
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
oldresume = 'exp/co-learning_ECAPA-TDNN_MCNN_Fbank80_ASP_emb192_2.0s/pretrained0.80.model'
newresume = 'exp/co-learning_ECAPA-TDNN_MCNN_Fbank80_ASP_emb192_2.0s/audionet_pretrained.pth'
# oldstatedictname = 'AVLBstate_dict'
newstatedictname = 'audiostate_dict'
statedict_pairs = [['speaker_encoder.torchfbank', 'feature.torchfbank'], 
                   ['speaker_encoder.conv1', 'AudioEncoder.conv1'],
                   ['speaker_encoder.bn1', 'AudioEncoder.bn1'],
                   ['speaker_encoder.layer', 'AudioEncoder.layer'],
                   ['speaker_encoder.attention', 'AudioDecoder.attention'],
                   ['speaker_encoder.bn5', 'AudioDecoder.bn1'],
                   ['speaker_encoder.fc6', 'AudioDecoder.fc1'],
                   ['speaker_encoder.bn6', 'AudioDecoder.bn2']] # old, new
statedict_notadd = []

print('loading old model from {}'.format(oldresume))
if not os.path.exists(oldresume):
    print('No pretrained model exists!')
else:
    ckpt = torch.load(oldresume)
    old_statedict = ckpt#[oldstatedictname]
    new_statedict = {}
    for i, pairs in enumerate(statedict_pairs):
        old_property, new_property = pairs[0], pairs[1]
        for k, v in old_statedict.items():
            new_k = 'module.' + k.replace(old_property, new_property)
            if old_property in k:
                new_statedict[new_k] = v
    audiocriterion = AAMsoftmax(n_class=5994, m=0.2, s=30, em_dim=192)
    # audiocriterion.weight.data = changelabelmap(ckpt['speaker_loss.weight'])
    audiocriterion.weight.data = ckpt['speaker_loss.weight']
    torch.save({'epoch': 0, 
                newstatedictname: new_statedict,
                'audiocriterion': audiocriterion},
                newresume)
print('saved new model from {}'.format(newresume))


