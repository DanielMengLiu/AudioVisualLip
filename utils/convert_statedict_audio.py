import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
oldresume = 'exp/ECAPA_Vox2_eer1.25_epoch110/net_110.pth'
newresume = 'exp/co-learning_ECAPA-TDNN_ResNet18_emb192_2.0s/audionet_pretrained.pth'
oldstatedictname = 'AVLBstate_dict'
newstatedictname = 'audiostate_dict'
statedict_pairs = [['Sync_AudioEncoder', 'AudioEncoder'], ['Sync_Decoder_A', 'AudioDecoder'], \
    ['Sync_AudioEncoder.torchfbank', 'feature.torchfbank'], ['Sync_Decoder_A.bn5', 'AudioDecoder.bn1'], \
    ['Sync_Decoder_A.fc6', 'AudioDecoder.fc1'], ['Sync_Decoder_A.bn6', 'AudioDecoder.bn2']] # old, new
statedict_notadd = ['AudioEncoder.torchfbank', 'AudioDecoder.bn5', 'AudioDecoder.fc6', 'AudioDecoder.bn6']

print('loading old model from {}'.format(oldresume))
if not os.path.exists(oldresume):
    print('No pretrained model exists!')
else:
    ckpt = torch.load(oldresume)
    old_statedict = ckpt[oldstatedictname]
    if oldstatedictname in ckpt.keys():
        new_statedict = {}
        for i, pairs in enumerate(statedict_pairs):
            old_property, new_property = pairs[0], pairs[1]
            for k, v in old_statedict.items():
                new_k = k.replace(old_property, new_property)
                if old_property in k:
                    new_statedict[new_k] = v
        wrongks = []
        for k in new_statedict.keys():
            for wrongk in statedict_notadd:
                if wrongk in k: wrongks.append(k)
        for wrongk in wrongks: del new_statedict[wrongk]
        torch.save({'epoch': 0, 
                    newstatedictname: new_statedict,
                    'audiocriterion': ckpt['audiocriterion']},
                    newresume)
    else:
        print('error: not find {} in old model!'.format(oldstatedictname))
print('saved new model from {}'.format(newresume))
        