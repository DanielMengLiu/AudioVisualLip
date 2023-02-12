import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
oldresume = 'exp/vox_cross-modal_ECAPA-TDNN_MCNN/visualnet_pretrained.pth'
newresume = 'exp/vox_cross-modal_ECAPA-TDNN_MCNN/visualnet_pretrained.pth'
oldstatedictname = 'visualstate_dict'
newstatedictname = 'visualstate_dict'
statedict_notadd = ['VisualDecoder']

print('loading old model from {}'.format(oldresume))
if not os.path.exists(oldresume):
    print('No pretrained model exists!')
else:
    ckpt = torch.load(oldresume)
    old_statedict = ckpt[oldstatedictname]
    if oldstatedictname in ckpt.keys():
        wrongks = []
        for k in old_statedict.keys():
            for wrongk in statedict_notadd:
                if wrongk in k: wrongks.append(k)
        for wrongk in wrongks: del old_statedict[wrongk]
        torch.save({'epoch': 0, 
                    newstatedictname: old_statedict},
                    newresume)
    else:
        print('error: not find {} in old model!'.format(oldstatedictname))
print('saved new model from {}'.format(newresume))
                