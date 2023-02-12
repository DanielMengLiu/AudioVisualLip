import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
oldresume = 'exp/ResNet18_face224/net_47.pth'
newresume = 'exp/ResNet18_face224/visualnet_pretrained.pth'
oldstatedictname = 'AVLBstate_dict'
newstatedictname = 'visualstate_dict'
statedict_pairs = [['Sync_VideoEncoder.resnet2D', 'VisualEncoder']] # old, new
statedict_notadd = ['Sync_VideoEncoder.fronted3D', 'Sync_VideoEncoder.tcn1D']

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
                    newstatedictname: new_statedict},
                    newresume)
    else:
        print('error: not find {} in old model!'.format(oldstatedictname))
print('saved new model from {}'.format(newresume))
        