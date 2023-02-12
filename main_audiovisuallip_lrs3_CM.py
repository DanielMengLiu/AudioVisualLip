from readline import append_history_file
import wandb, shutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os, yaml, pprint, warnings
from yaml import Loader as CLoader
from tqdm import tqdm
import torch.multiprocessing as mp
from multiprocessing import Process, Manager
from torch.utils.data import DataLoader
import datasets_new as datasets
from models.loss import AAMsoftmax, KLDivergenceLoss, ValueLoss, CoregularizationLoss
from models.network_new import AudioModel, VisualModel, AudioVisualModel
from utils.eval_metrics_new import *


################# GLOBAL CONFIGURATION ##############################
MODE  = 'test'           # train | test | finetune
#####################################################################

SEED  = 2022              # fixed random seed for fair comparison
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings('ignore')

with open('./conf/savedsolution_audiovisuallip_lrs3_CM.yaml', 'r') as f:
    OPTS = yaml.load(f, Loader=CLoader)
# shutil.copyfile('./conf/config_audiovisuallip_lrs3.yaml', './conf/savedsolution_audiovisuallip_lrs3_CM_pretrained.yaml')

# Basic functions for train and test phase
def get_audiomodel(model):
    return AudioModel(model, C=512)

def get_visualmodel(model):
    return VisualModel(model)

def get_audiovisualmodel(model):
    return AudioVisualModel(model)

class Trainer(object):
    def __init__(self):
        self.stageopts = OPTS[MODE]
        self.audiomodelname = self.stageopts['audiomodel']
        self.visualmodelname = self.stageopts['visualmodel']
        self.audiovisualmodelname = self.stageopts['audiovisualmodel']
        self.audiomodelopts = OPTS[self.audiomodelname]
        self.visualmodelopts = OPTS[self.visualmodelname]
        self.audioresume = self.stageopts['audioresume']
        self.visualresume = self.stageopts['visualresume']
        self.audiovisualresume = self.stageopts['audiovisualresume']
        self.audioemb_dim = self.audiomodelopts['embedding_dim']
        self.visualemb_dim = self.visualmodelopts['embedding_dim']

        # neural network
        audiomodel = get_audiomodel(self.audiomodelname)
        visualmodel = get_visualmodel(self.visualmodelname)
        audiovisualmodel = get_audiovisualmodel(self.audiovisualmodelname)
        print('audio model parameters_count: %.2fM' % (sum(p.numel() for p in audiomodel.parameters() if p.requires_grad)/1e6))
        print('visual model parameters_count: %.2fM' % (sum(p.numel() for p in visualmodel.parameters() if p.requires_grad)/1e6))
        print('visual model parameters_count: %.2fM' % (sum(p.numel() for p in audiovisualmodel.parameters() if p.requires_grad)/1e6))
        
        device_ids, device_num = self.stageopts['gpus'], len(self.stageopts['gpus'])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in device_ids])
        self.device_ids = [x for x in range(len(device_ids))]
        self.device = torch.device('cuda') # : +str(device_ids[0])
        self.audiomodel = torch.nn.DataParallel(audiomodel.to(self.device), device_ids=self.device_ids)
        self.visualmodel = torch.nn.DataParallel(visualmodel.to(self.device), device_ids=self.device_ids)
        self.audiovisualmodel = torch.nn.DataParallel(audiovisualmodel.to(self.device), device_ids=self.device_ids)
        
        self.featureopts = OPTS[self.audiomodelopts['feature']]
        self.dataopts = {}
        self.dataopts = {**{'seconds':self.stageopts['seconds']}, **{'sample_rate':self.featureopts['sample_rate']} }
        for aug in self.stageopts['augmentation'].keys():
            self.dataopts = {**self.dataopts, **{aug:self.stageopts['augmentation'][aug]}}
        for traindata in ['train_manifest', 'train_audiodir', 'train_videodir', 'musan_path', 'rir_path']: # for train and aug 
            self.dataopts = {**self.dataopts, **{traindata:OPTS[traindata]}}
        for valdata in ['test_trial', 'test_audiodir', 'test_videodir']: # for val
            self.dataopts = {**self.dataopts, **{valdata:OPTS['test_lrs3'][valdata]}}

        self.trainset = datasets.AudioVisualLipTrainset(self.dataopts)
        self.trainloader = DataLoader(self.trainset, shuffle=True, batch_size=self.stageopts['batchsize'], num_workers=4*device_num, drop_last=True)

        self.audiocriterion = AAMsoftmax(n_class=self.trainset.n_spk, m=self.stageopts['margins'][1], s=self.stageopts['scale'], em_dim=self.audioemb_dim).to(self.device) 
        self.visualcriterion = AAMsoftmax(n_class=self.trainset.n_spk, m=self.stageopts['margins'][1], s=self.stageopts['scale'], em_dim=self.audioemb_dim).to(self.device)
        self.transaudiocriterion = AAMsoftmax(n_class=self.trainset.n_spk, m=self.stageopts['margins'][1], s=self.stageopts['scale'], em_dim=self.audioemb_dim).to(self.device)  
        self.transvisualcriterion = AAMsoftmax(n_class=self.trainset.n_spk, m=self.stageopts['margins'][1], s=self.stageopts['scale'], em_dim=self.audioemb_dim).to(self.device)
        self.transaudioalignment = torch.nn.CosineEmbeddingLoss().to(self.device) 
        self.transvisualalignment = torch.nn.CosineEmbeddingLoss().to(self.device)
        
        param_groups = [{'params': self.audiomodel.parameters()}, 
                        {'params': self.visualmodel.parameters()}, 
                        {'params': self.audiovisualmodel.parameters()},
                        {'params': self.audiocriterion.parameters()},
                        {'params': self.visualcriterion.parameters()},
                        {'params': self.transaudiocriterion.parameters()},
                        {'params': self.transvisualcriterion.parameters()},
                        # {'params': self.transaudioalignment.parameters()},
                        # {'params': self.transvisualalignment.parameters()},
                        ]
        
        if self.stageopts['optimizer'] == 'sgd':
            self.optimopts = OPTS['sgd']
            self.optim = optim.SGD(param_groups, self.optimopts['init_lr'], nesterov=self.optimopts['nesterov'], momentum = self.optimopts['momentum'], weight_decay = self.optimopts['weight_decay'])
        elif self.stageopts['optimizer'] == 'adam':
            self.optimopts = OPTS['adam']
            self.optim = optim.Adam(param_groups, lr=self.optimopts['init_lr'] , weight_decay=self.optimopts['weight_decay'])
        if self.stageopts['lr_scheduler'] == 'steplr':
            #self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=3, gamma=0.5)
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10,15], gamma=0.1)
        elif self.stageopts['lr_scheduler'] == 'cycliclr':
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optim,cycle_momentum=False,base_lr=0.000001,max_lr=0.001,step_size_up=2000,step_size_down=2000)
        
        self.aeers = []
        self.adcfs = []
        self.veers = []
        self.vdcfs = []
        self.current_epoch, self.epochs = 0, self.stageopts['epochs']
        if self.audioresume == 'None' or self.visualresume == 'None':
            self.exp = '_'.join(['lrs3', 'cross-modal', self.audiomodelname, self.visualmodelname, 'noisespec1blocks'])
        else:
            self.exp = self.audioresume.split('/')[1]

        # continue training
        if not ((self.audioresume == 'None' and not os.path.exists(self.audioresume)) and (self.visualresume == 'None' and not os.path.exists(self.visualresume)) and (self.audiovisualresume == 'None' and not os.path.exists(self.audiovisualresume))):
            self._load()
        else:
            print('Train from scratch: there is no specified audio resume.')

        self.wandb = self.stageopts['wandb']
        if self.wandb != False:
            config = {
                "learning_rate": self.optimopts['init_lr'],
                "epochs": self.epochs,
                "batch_size": self.stageopts['batchsize']
            }
            wandb.init(project="Short-Short-New", config=config)

    def _plan(self):
        valuerate = max(((self.current_epoch - 1)/(20 - 1)) * (0.8 - 1) + 1, 0.8)
        return valuerate

    def _load(self):
        print('loading audio model from {}'.format(self.audioresume))
        print('loading visual model from {}'.format(self.visualresume))
        print('loading audiovisual model from {}'.format(self.audiovisualresume))

        if not os.path.exists(self.audioresume):
            print('No pretrained audio model exists!')
        else:
            audiockpt = torch.load(self.audioresume)
            if 'audiostate_dict' in audiockpt.keys():
                self.audiomodel.load_state_dict(audiockpt['audiostate_dict'], strict=False)         
            if 'audiocriterion' in audiockpt.keys():
                self.audiocriterion = audiockpt['audiocriterion'] 
        if not os.path.exists(self.visualresume):
            print('No pretrained visual model exists!')
        else:
            visualckpt = torch.load(self.visualresume)
            if 'visualstate_dict' in visualckpt.keys():
                self.visualmodel.load_state_dict(visualckpt['visualstate_dict'], strict=False)
            if 'visualcriterion' in visualckpt.keys():
                self.visualcriterion = visualckpt['visualcriterion']
        if not os.path.exists(self.audiovisualresume):
            print('No pretrained audiovisual model exists!')
        else:
            audiovisualckpt = torch.load(self.audiovisualresume)
            if 'audiovisualstate_dict' in audiovisualckpt.keys():
                self.audiovisualmodel.load_state_dict(audiovisualckpt['audiovisualstate_dict']) #, strict=False
            if 'transaudiocriterion' in audiovisualckpt.keys():
                self.transaudiocriterion = audiovisualckpt['transaudiocriterion']
            if 'transvisualcriterion' in audiovisualckpt.keys():
                self.transvisualcriterion = audiovisualckpt['transvisualcriterion']
            if 'epoch' in audiovisualckpt.keys():
                self.current_epoch = audiovisualckpt['epoch']
            if 'optimizer' in audiovisualckpt.keys():
                self.optim.load_state_dict(audiovisualckpt['optimizer'])

                
    def _train(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            self._train_epoch()
            self.lr_scheduler.step()

    def _train_epoch(self):
        self.audiomodel.train()
        self.visualmodel.train()
        self.audiovisualmodel.train()
        self.audiocriterion.train()
        self.visualcriterion.train()
        self.transaudiocriterion.train()
        self.transvisualcriterion.train()
        # self.transaudioalignment.train()
        # self.transvisualalignment.train()
        audiosum_loss, audiosum_samples, audiocorrect = 0, 0, 0
        videosum_loss, videosum_samples, videocorrect = 0, 0, 0
        transaudiosum_loss, transvideosum_loss = 0, 0
        progress_bar = tqdm(self.trainloader)
        
        for batch_idx, (audiofeats, videofeats, targets_label) in enumerate(progress_bar):
            self.optim.zero_grad()
            audiofeats = audiofeats.to(self.device)
            videofeats = videofeats.to(self.device)
            targets_label = targets_label.to(self.device)
            similar_label = torch.ones(targets_label.shape[0]).to(self.device)

            frame_audio, emb_audio = self.audiomodel(audiofeats, aug=True) #
            frame_video, emb_video = self.visualmodel(videofeats)
            emb_transaudio, emb_transvideo = self.audiovisualmodel(frame_audio, frame_video)
            
            aloss, audiologits = self.audiocriterion(emb_audio, targets_label)
            vloss, videologits = self.visualcriterion(emb_video, targets_label)
            traloss, tralogits = self.transaudiocriterion(emb_transaudio, targets_label)
            trvloss, trvlogits = self.transvisualcriterion(emb_transvideo, targets_label)
            # traalignloss = self.transaudioalignment(emb_transaudio, emb_audio, similar_label)
            # trvalignloss = self.transvisualalignment(emb_transvideo, emb_video, similar_label)
            
            #avloss, aloss, vloss, traloss, trvloss, traalignloss, trvalignloss = CoregularizationLoss(aloss, vloss, traloss, trvloss, traalignloss, trvalignloss) #          
            avloss = aloss + vloss + traloss + trvloss
            avloss.backward()

            audiosum_samples += len(audiofeats)
            videosum_samples += len(videofeats)
            _, audioprediction = torch.max(audiologits, dim=1)
            _, videoprediction = torch.max(videologits, dim=1)
            audiocorrect += (audioprediction == targets_label).sum().item()
            videocorrect += (videoprediction == targets_label).sum().item()

            # if self.current_epoch == 1 and batch_idx < 2000:
            #     lr_scale = min(1., float(batch_idx + 1) / float(2000))
            #     for _, pg in enumerate(self.optim.param_groups):
            #         pg['lr'] = lr_scale * 0.001

            self.optim.step()
         
            audiosum_loss += aloss.item() * len(targets_label)
            videosum_loss += vloss.item() * len(targets_label)
            transaudiosum_loss += traloss.item() * len(targets_label)
            transvideosum_loss += trvloss.item() * len(targets_label)
            progress_bar.set_description(
                    'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] audioLoss: {:.4f} audioAcc: {:.4f}% videoLoss: {:.4f} videoAcc: {:.4f}%' #  
                    .format(self.current_epoch, batch_idx + 1,
                    len(self.trainloader), 100. * (batch_idx + 1) / len(self.trainloader),
                    audiosum_loss / audiosum_samples, 100. * audiocorrect / audiosum_samples,
                    videosum_loss / videosum_samples, 100. * videocorrect / videosum_samples))
            if self.wandb != False:
                wandb.log({"audioLoss":audiosum_loss/audiosum_samples,
                           "videoLoss":videosum_loss/videosum_samples,
                           "TransaudioLoss":transaudiosum_loss/audiosum_samples,
                           "TransvideoLoss":transvideosum_loss/videosum_samples,
                           "lr": self.optim.state_dict()['param_groups'][0]['lr'],
                           })
        self._save('exp/{}/net_{}.pth'.format(self.exp, self.current_epoch))
        # flexible log
        interval_val = 1
        if self.current_epoch % interval_val == 0:
            aeer, aminDCF, veer, vminDCF = self._eval_network(stage='val', num_workers=1)
            self.aeers.append(aeer)         
            self.adcfs.append(aminDCF)    
            self.veers.append(veer)         
            self.vdcfs.append(vminDCF)    
            with open('log/'+self.exp+'-training.log', "a+") as score_file:   
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, aEER %2.2f%%, aminDCF %2.4f, abestEER %2.2f%%, abestminDCF %2.4f\n"  \
                                %(self.current_epoch, self.optim.state_dict()['param_groups'][0]['lr'], audiosum_loss / audiosum_samples, \
                                100. * audiocorrect / audiosum_samples, self.aeers[-1], self.adcfs[-1], min(self.aeers), min(self.adcfs)))
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, vEER %2.2f%%, vminDCF %2.4f, vbestEER %2.2f%%, vbestminDCF %2.4f\n"  \
                                %(self.current_epoch, self.optim.state_dict()['param_groups'][0]['lr'], videosum_loss / videosum_samples, \
                                100. * videocorrect / videosum_samples, self.veers[-1], self.vdcfs[-1], min(self.veers), min(self.vdcfs)))
                score_file.flush()   
        
    def _save(self, modelpath):
        torch.save({'audiostate_dict': self.audiomodel.state_dict(),
                    'audiocriterion': self.audiocriterion},
                    modelpath.replace('net_','audionet_'))
        torch.save({'visualstate_dict': self.visualmodel.state_dict(),
                    'visualcriterion': self.visualcriterion},
                    modelpath.replace('net_','visualnet_'))
        torch.save({'epoch': self.current_epoch,
                    'audiovisualstate_dict': self.audiovisualmodel.state_dict(),
                    'transaudiocriterion': self.transaudiocriterion,
                    'transvisualcriterion': self.visualcriterion,
                    # 'transaudioalignment': self.transaudioalignment,
                    # 'transvisualalignment': self.transvisualalignment,
                    'optimizer': self.optim.state_dict()},
                    modelpath.replace('net_','audiovisualnet_'))

    def _extract_embedding(self, stage, filelist):  # stage='val'
        testset = datasets.AudioVisualLipTestset(self.dataopts, filelist, stage='val')
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

        audiotestmodel = self.audiomodel.module
        visualtestmodel = self.visualmodel.module
        audiotestmodel = audiotestmodel.to(self.device)
        visualtestmodel = visualtestmodel.to(self.device)
        audiotestmodel.eval()
        visualtestmodel.eval()
        # print('Extracting test embeddings for {}: '.format(dataset))
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        os.makedirs(emb_dir, exist_ok = True)
        with torch.no_grad():
            for audiosignal_global, audiosignal_local, videosignal_global, videosignal_local, utt in tqdm(testloader):
                utt = utt[0].split('.')[0]
                spk_dir = os.path.join(emb_dir, os.path.dirname(utt))
                os.makedirs(spk_dir, exist_ok=True)

                audiosignal_global = audiosignal_global.squeeze(0).to(self.device)
                audiosignal_local = audiosignal_local.squeeze(0).to(self.device)
                videosignal_global = videosignal_global.to(self.device)
                videosignal_local = videosignal_local.squeeze(0).to(self.device)
                
                audioframe_global, audioembedding_global = audiotestmodel(audiosignal_global)
                audioframe_local, audioembedding_local = audiotestmodel(audiosignal_local)
                videoframe_global, videoembedding_global = visualtestmodel(videosignal_global)
                videoframe_local, videoembedding_local = visualtestmodel(videosignal_local)
                
                audioembedding_global, audioembedding_local = audioembedding_global.cpu().numpy(), audioembedding_local.cpu().numpy()
                videoembedding_global, videoembedding_local = videoembedding_global.cpu().numpy(), videoembedding_local.cpu().numpy()
                np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_a'), [audioembedding_global, audioembedding_local])
                np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_v'), [videoembedding_global, videoembedding_local])
        del audiotestmodel
        del visualtestmodel
        torch.cuda.empty_cache()

    def _score_embedding(self, stage, trials, ascore_dict, vscore_dict):  
        # GRID scoring for global and local embeddings and save score in score/MODELDIR/score_TRIAL.txt
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        for line in trials:
            aembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][0])
            aembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][1])
            aembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][0])
            aembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][1])
            aembedding_11, aembedding_12, aembedding_21, aembedding_22 = F.normalize(aembedding_11),F.normalize(aembedding_12),F.normalize(aembedding_21),F.normalize(aembedding_22)
            vembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][0])
            vembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][1])
            vembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][0])
            vembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][1])
            vembedding_11, vembedding_12, vembedding_21, vembedding_22 = F.normalize(vembedding_11),F.normalize(vembedding_12),F.normalize(vembedding_21),F.normalize(vembedding_22)

            # Compute the a scores
            ascore_1 = torch.mean(torch.matmul(aembedding_11, aembedding_21.T)) # higher is positive
            ascore_2 = torch.mean(torch.matmul(aembedding_12, aembedding_22.T))
            ascore = (ascore_1 + ascore_2) / 2
            ascore = ascore.detach().cpu().numpy()
            ascore_dict[line] = [ascore, int(line.split()[0])] # score, label
            # Compute the v scores
            vscore_1 = torch.mean(torch.matmul(vembedding_11, vembedding_21.T)) # higher is positive
            vscore_2 = torch.mean(torch.matmul(vembedding_12, vembedding_22.T))
            vscore = (vscore_1 + vscore_2) / 2
            vscore = vscore.detach().cpu().numpy()
            vscore_dict[line] = [vscore, int(line.split()[0])] # score, label
            
    def _eval_network(self, stage, num_workers=1):
        # prepare for multiprocessing of embedding extraction
        files, utts = [], []
        trial = self.dataopts['test_trial']
        lines = open(trial).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()   
        part = list(range(0, len(setfiles)+1, int(len(setfiles)//num_workers)))
        part[-1] = len(setfiles)
        utts = [setfiles[part[i]:part[i+1]] for i in range(num_workers)]

        self.audiomodel.cpu()
        self.visualmodel.cpu()
        torch.cuda.empty_cache()
        if num_workers == 1:
            self._extract_embedding(stage, utts[0])
        else:
            # extract embeddings and save in exp/MODELDIR/test_emb/
            args = [(stage, utts[i]) for i in range(num_workers)]
            ctx = mp.get_context("spawn")
            jobs = [ctx.Process(target=self._extract_embedding, args=(a)) for a in args]
            for j in jobs: j.start()
            for j in jobs: j.join()  
        self.audiomodel = self.audiomodel.to(self.device)
        self.visualmodel = self.visualmodel.to(self.device)

        num_workers = 40
        ascore_dict = Manager().dict()
        vscore_dict = Manager().dict()
        part = list(range(0, len(lines)+1, int(len(lines)//num_workers)))
        part[-1] = len(lines)
        trials = [lines[part[i]:part[i+1]] for i in range(num_workers)]
        args = [(stage, trials[i], ascore_dict, vscore_dict) for i in range(num_workers)]
        jobs = [Process(target=self._score_embedding, args=(a)) for a in args]
        for j in jobs: j.start()
        for j in jobs: j.join()  

        ascores, alabels = [], []
        for line in lines:
            ascores.append(ascore_dict[line][0])
            alabels.append(ascore_dict[line][1])
        # Coumpute EER and minDCF
        aEER = tuneThresholdfromScore(ascores, alabels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(ascores, alabels)
        aminDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        vscores, vlabels = [], []
        for line in lines:
            vscores.append(vscore_dict[line][0])
            vlabels.append(vscore_dict[line][1])
        # Coumpute EER and minDCF
        vEER = tuneThresholdfromScore(vscores, vlabels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(vscores, vlabels)
        vminDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("aEER: {:.6f}%, aminDCF: {:.6f}, vEER: {:.6f}%, vminDCF: {:.6f}".format(aEER, aminDCF, vEER, vminDCF))
        return aEER, aminDCF, vEER, vminDCF

    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)
 
    def __call__(self):
        print("[Model is saved in: {}]".format(self.exp))
        os.makedirs('exp/{}'.format(self.exp), exist_ok = True)
        self._train()


class Tester(object):
    def __init__(self):
        self.stageopts = OPTS[MODE]
        self.audiomodelname = self.stageopts['audiomodel']
        self.visualmodelname = self.stageopts['visualmodel']
        self.audiovisualmodelname = self.stageopts['audiovisualmodel']
        self.audiomodelopts = OPTS[self.audiomodelname]
        self.audioemb_dim = self.audiomodelopts['embedding_dim']
        self.audioresume = self.stageopts['audioresume']
        self.visualresume = self.stageopts['visualresume']
        self.audiovisualresume = self.stageopts['audiovisualresume']
        self.audiovisualembedding = self.stageopts['audiovisualembedding']
        self.type = self.stageopts['type']
        self.exp = self.audioresume.split('/')[1]

        self.gpus = self.stageopts['gpus']
        self.device_ids = [x for x in range(len(self.gpus))]

        self.featureopts = OPTS[self.audiomodelopts['feature']]
        self.dataopts = {}
        for data in ['train_manifest', 'train_audiodir', 'train_videodir', 'cohort_manifest']: # for submean, cohort, and test
            self.dataopts = {**self.dataopts, **{data:OPTS[data]}}
        for testdata in ['test_trial', 'test_audiodir', 'test_videodir']: # for test
            self.dataopts = {**self.dataopts, **{testdata:OPTS[self.stageopts['data']][testdata]}}
        self.embeds = Manager().dict()
        
    def _extract_embedding(self, stage, filelist, gpu):  # test | cohort | submean
        if self.type == 'audio':
            testset = datasets.AudioTestset(self.dataopts, filelist, stage)
        elif self.type == 'visual':
            testset = datasets.VisualLipTestset(self.dataopts, filelist, stage)
        elif self.type == 'audiovisual':
            testset = datasets.AudioVisualLipTestset(self.dataopts, filelist, stage)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device('cuda')
        if self.type == 'audio':
            testmodel = get_audiomodel(self.audiomodelname)
        elif self.type == 'visual':
            testmodel = get_visualmodel(self.visualmodelname)
        elif self.type == 'audiovisual':
            audiotestmodel = get_audiomodel(self.audiomodelname)
            visualtestmodel = get_visualmodel(self.visualmodelname)
            avtestmodel = get_audiovisualmodel(self.audiovisualmodelname)
        
        if self.type == 'audio' or self.type == 'visual':
            testmodel = torch.nn.DataParallel(testmodel.to(device), device_ids = [0])            
        elif self.type == 'audiovisual':
            audiotestmodel = torch.nn.DataParallel(audiotestmodel.to(device), device_ids = [0])
            visualtestmodel = torch.nn.DataParallel(visualtestmodel.to(device), device_ids = [0])
            avtestmodel = torch.nn.DataParallel(avtestmodel.to(device), device_ids = [0])

        audiockpt = torch.load(self.audioresume)
        visualckpt = torch.load(self.visualresume)
        audiovisualckpt = torch.load(self.audiovisualresume)
        if self.type == 'audio':
            if 'audiostate_dict' in audiockpt.keys():
                testmodel.load_state_dict(audiockpt['audiostate_dict'])
        elif self.type == 'visual':
            if 'visualstate_dict' in visualckpt.keys():
                testmodel.load_state_dict(visualckpt['visualstate_dict'], strict=False)                
        elif self.type == 'audiovisual':
            if 'audiostate_dict' in audiockpt.keys():
                audiotestmodel.load_state_dict(audiockpt['audiostate_dict'], strict=False)
            if 'visualstate_dict' in visualckpt.keys():
                visualtestmodel.load_state_dict(visualckpt['visualstate_dict'], strict=False)      
            if 'audiovisualstate_dict' in audiovisualckpt.keys():
                avtestmodel.load_state_dict(audiovisualckpt['audiovisualstate_dict'], strict=False)
                
        if self.type == 'audio' or self.type == 'visual': testmodel.module.eval()
        elif self.type == 'audiovisual':
            audiotestmodel.module.eval()
            visualtestmodel.module.eval()
            avtestmodel.module.eval()
        # print('Extracting test embeddings for {}: '.format(dataset))
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        os.makedirs(emb_dir, exist_ok = True)
        with torch.no_grad():
            if self.type == 'audio' or self.type == 'visual':
                for signal_global, signal_local, utt in tqdm(testloader):
                    utt = utt[0].split('.')[0]
                    spk_dir = os.path.join(emb_dir, os.path.dirname(utt))
                    os.makedirs(spk_dir, exist_ok=True)

                    signal_global = signal_global.squeeze(0).to(device)
                    signal_local = signal_local.squeeze(0).to(device)

                    _, embedding_global = testmodel(signal_global)
                    _, embedding_local = testmodel(signal_local)

                    embedding_global = embedding_global.cpu().numpy()
                    embedding_local = embedding_local.cpu().numpy()
                    np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)), [embedding_global, embedding_local])
            elif self.type == 'audiovisual':
                for audiosignal_global, audiosignal_local, videosignal_global, videosignal_local, utt in tqdm(testloader):
                    utt = utt[0].split('.')[0]
                    spk_dir = os.path.join(emb_dir, os.path.dirname(utt))
                    os.makedirs(spk_dir, exist_ok=True)

                    audiosignal_global = audiosignal_global.squeeze(0).to(device)
                    audiosignal_local = audiosignal_local.squeeze(0).to(device)
                    videosignal_global = videosignal_global.to(device)
                    videosignal_local = videosignal_local.squeeze(0).to(device)
                    
                    audioframe_global, audioembedding_global = audiotestmodel(audiosignal_global)
                    audioframe_local, audioembedding_local = audiotestmodel(audiosignal_local)
                    videoframe_global, videoembedding_global = visualtestmodel(videosignal_global)
                    videoframe_local, videoembedding_local = visualtestmodel(videosignal_local)
                    transaudioembedding_global, transvideoembedding_global = avtestmodel(audioframe_global, videoframe_global)
                    transaudioembedding_local, transvideoembedding_local = avtestmodel(audioframe_local, videoframe_local)
                    
                    audioembedding_global, audioembedding_local = audioembedding_global.cpu().numpy(), audioembedding_local.cpu().numpy()
                    videoembedding_global, videoembedding_local = videoembedding_global.cpu().numpy(), videoembedding_local.cpu().numpy()
                    transaudioembedding_global, transaudioembedding_local = transaudioembedding_global.cpu().numpy(), transaudioembedding_local.cpu().numpy()
                    transvideoembedding_global, transvideoembedding_local = transvideoembedding_global.cpu().numpy(), transvideoembedding_local.cpu().numpy()
                    if self.stageopts['embed_extract'] == 'offline':
                        np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_a'), [audioembedding_global, audioembedding_local])
                        np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_v'), [videoembedding_global, videoembedding_local])
                        np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_a\''), [transaudioembedding_global, transaudioembedding_local])
                        np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt)+'_v\''), [transvideoembedding_global, transvideoembedding_local])
                    elif self.stageopts['embed_extract'] == 'online':
                        self.embeds[os.path.join(spk_dir, os.path.basename(utt)+'_a')] = [audioembedding_global, audioembedding_local]
                        self.embeds[os.path.join(spk_dir, os.path.basename(utt)+'_v')] = [videoembedding_global, videoembedding_local]
                        self.embeds[os.path.join(spk_dir, os.path.basename(utt)+'_a\'')] = [transaudioembedding_global, transaudioembedding_local]
                        self.embeds[os.path.join(spk_dir, os.path.basename(utt)+'_v\'')] = [transvideoembedding_global, transvideoembedding_local]

    def _score_embedding(self, stage, trials, score_dict):  
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        for line in trials:
            if self.type == 'audio' or self.type == 'visual':
                embedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
                embedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
                embedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
                embedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
                embedding_11, embedding_12, embedding_21, embedding_22 = F.normalize(embedding_11),F.normalize(embedding_12),F.normalize(embedding_21),F.normalize(embedding_22)
            elif self.type == 'audiovisual':
                if self.stageopts['embed_extract'] == 'offline':
                    aembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][0])
                    aembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][1])
                    aembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][0])
                    aembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a.npz'), allow_pickle=True)['arr_0'][1])
                    vembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][0])
                    vembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][1])
                    vembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][0])
                    vembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v.npz'), allow_pickle=True)['arr_0'][1])
                    transaembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a\'.npz'), allow_pickle=True)['arr_0'][0])
                    transaembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a\'.npz'), allow_pickle=True)['arr_0'][1])
                    transaembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a\'.npz'), allow_pickle=True)['arr_0'][0])
                    transaembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a\'.npz'), allow_pickle=True)['arr_0'][1])
                    transvembedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v\'.npz'), allow_pickle=True)['arr_0'][0])
                    transvembedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v\'.npz'), allow_pickle=True)['arr_0'][1])
                    transvembedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v\'.npz'), allow_pickle=True)['arr_0'][0])
                    transvembedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v\'.npz'), allow_pickle=True)['arr_0'][1])                                
                elif self.stageopts['embed_extract'] == 'online':
                    aembedding_11 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a')][0])
                    aembedding_12 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a')][1])
                    aembedding_21 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a')][0])
                    aembedding_22 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a')][1])
                    vembedding_11 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v')][0])
                    vembedding_12 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v')][1])
                    vembedding_21 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v')][0])
                    vembedding_22 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v')][1])
                    transaembedding_11 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a\'')][0])
                    transaembedding_12 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_a\'')][1])
                    transaembedding_21 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a\'')][0])
                    transaembedding_22 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_a\'')][1])
                    transvembedding_11 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v\'')][0])
                    transvembedding_12 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-2].split('.')[0]+'_v\'')][1])
                    transvembedding_21 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v\'')][0])
                    transvembedding_22 = torch.FloatTensor(self.embeds[os.path.join(emb_dir, line.split()[-1].split('.')[0]+'_v\'')][1])                    
                aembedding_11, aembedding_12, aembedding_21, aembedding_22 = F.normalize(aembedding_11),F.normalize(aembedding_12),F.normalize(aembedding_21),F.normalize(aembedding_22)
                vembedding_11, vembedding_12, vembedding_21, vembedding_22 = F.normalize(vembedding_11),F.normalize(vembedding_12),F.normalize(vembedding_21),F.normalize(vembedding_22)
                transaembedding_11, transaembedding_12, transaembedding_21, transaembedding_22 = F.normalize(transaembedding_11),F.normalize(transaembedding_12),F.normalize(transaembedding_21),F.normalize(transaembedding_22)
                transvembedding_11, transvembedding_12, transvembedding_21, transvembedding_22 = F.normalize(transvembedding_11),F.normalize(transvembedding_12),F.normalize(transvembedding_21),F.normalize(transvembedding_22)
                embedding_11, embedding_12, embedding_21, embedding_22 = [], [], [], []
                for x in self.audiovisualembedding:
                    if x == 'a': 
                        embedding_11.append(aembedding_11) 
                        embedding_12.append(aembedding_12)
                        embedding_21.append(aembedding_21)
                        embedding_22.append(aembedding_22)
                    if x == 'v':
                        embedding_11.append(vembedding_11) 
                        embedding_12.append(vembedding_12)
                        embedding_21.append(vembedding_21)
                        embedding_22.append(vembedding_22)
                    if x == 'transa':
                        embedding_11.append(transaembedding_11) 
                        embedding_12.append(transaembedding_12)
                        embedding_21.append(transaembedding_21)
                        embedding_22.append(transaembedding_22)
                    if x == 'transv':
                        embedding_11.append(transvembedding_11) 
                        embedding_12.append(transvembedding_12)
                        embedding_21.append(transvembedding_21)
                        embedding_22.append(transvembedding_22)
                embedding_11, embedding_12, embedding_21, embedding_22 = torch.cat(embedding_11, dim=-1),torch.cat(embedding_12, dim=-1),torch.cat(embedding_21, dim=-1),torch.cat(embedding_22, dim=-1)
                
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            if len(line.split()) == 3:
                score_dict[line] = [score, int(line.split()[0])]
            elif len(line.split()) == 2:
                score_dict[line] = [score]

    def _eval_network(self, stage, num_workers=1):
        # prepare for multiprocessing of embedding extraction
        files, utts = [], []
        if stage == 'test':
            trial = self.dataopts['test_trial']
            lines = open(trial).read().splitlines()
            for line in lines:
                files.append(line.split()[-2])
                files.append(line.split()[-1])
            setfiles = list(set(files))
            setfiles.sort()   
            part = list(range(0, len(setfiles)+1, int(len(setfiles)//num_workers)))
            part[-1] = len(setfiles)
            utts = [setfiles[part[i]:part[i+1]] for i in range(num_workers)]
        elif stage == 'cohort':
            trial = self.dataopts['cohort_manifest']
            lines = open(trial).read().splitlines()
            for line in lines:
                files.append(line)
        setfiles = list(set(files))
        setfiles.sort()   
        part = list(range(0, len(setfiles)+1, int(len(setfiles)//num_workers)))
        part[-1] = len(setfiles)
        utts = [setfiles[part[i]:part[i+1]] for i in range(num_workers)]

        print('loading audio model from {}'.format(self.audioresume))
        print('loading visual model from {}'.format(self.visualresume))
        print('loading audio-visual model from {}'.format(self.audiovisualresume))
        if num_workers == 1:
            self._extract_embedding(stage, utts[0], 0)
        else:
            # extract embeddings and save in exp/MODELDIR/test_emb/
            args = [(stage, utts[i], self.gpus[i%len(self.gpus)]) for i in range(num_workers)]
            jobs = [Process(target=self._extract_embedding, args=(a)) for a in args]
            for j in jobs: j.start()
            for j in jobs: j.join()
        
        if stage == 'cohort':
            print('finished') 
            return

        num_workers = 40
        score_dict = Manager().dict()
        part = list(range(0, len(lines)+1, int(len(lines)//num_workers)))
        part[-1] = len(lines)
        trials = [lines[part[i]:part[i+1]] for i in range(num_workers)]
        args = [(stage, trials[i], score_dict) for i in range(num_workers)]
        jobs = [Process(target=self._score_embedding, args=(a)) for a in args]
        for j in jobs: j.start()
        for j in jobs: j.join()  

        # GRID scoring for global and local embeddings and save score in score/MODELDIR/score_TRIAL.txt
        scores, labels  = [], []
        score_fname = os.path.join('score', self.exp, 'score_'+os.path.basename(trial))
        os.makedirs(os.path.join('score', self.exp), exist_ok = True)
        with open(score_fname, 'w+') as fout:
            for line in lines:
                scores.append(score_dict[line][0])
                if len(line.split()) == 3:
                    labels.append(score_dict[line][1])
                if self.stageopts['write_score']:
                    if len(line.split()) == 3:
                        fout.write('{} {} {:.5f} {}\n'.format(line.split()[-2], line.split()[-1], score_dict[line][0], line.split()[0]))
                    elif len(line.split()) == 2:
                        fout.write('{} {} {:.5f}\n'.format(line.split()[-2], line.split()[-1], score_dict[line][0])) 
    
        if len(line.split()) == 3:                
            # Coumpute EER and minDCF
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
            print("EER: {:.6f}%, minDCF: {:.6f}".format(EER, minDCF))
            
            # abs_diffs = np.abs(np.array(fnrs) - np.array(fprs))
            # min_index = np.argmin(abs_diffs)
            # threshold = thresholds[min_index]
            # with open('misclassification_v.txt', 'w') as fh:  
            #     fh.write(str(threshold)+'\n') 
            #     for (i, (k, v)) in enumerate(zip(labels, scores)):
            #         if (v < threshold and k == 1) \
            #             or (v >= threshold and k == 0):
            #             fh.write('{} {} {}\n'.format(i, k, v))        

            # with open('detail_v.txt', 'w') as fh:  
            #     for (i, (k, v)) in enumerate(zip(labels, scores)):
            #         if (v < threshold and k == 1) \
            #             or (v >= threshold and k == 0):
            #             fh.write('{} {} {}\n'.format(i, k, v))
            #         else:
            #             fh.write('{} {} {}\n'.format(i, -1, v))  


    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)


if __name__ == '__main__':
    if MODE == 'train' or MODE == 'finetune':
        trainer = Trainer()
        trainer()
    # voxceleb dataset test
    elif MODE == 'test': 
        tester = Tester()
        
        tester._eval_network(stage='test', num_workers=12)

    # elif MODE == 'extract_submean_embedding':
    #     trainer.extract_submeanembedding()    
    # elif MODE == 'featurefusion':
    #     eer, minDCF = utils.eer_cos_featurefusion(trainer.log_time, dataset='lrs3', trial=trainer.data_opts['test_lrs3']['test_trial_lrs3'])
    #     print("EER: {:.6f}%, minDCF: {:.6f}".format(eer * 100, minDCF))
    # elif MODE == 'scorefusion':
    #     eer, minDCF = utils.eer_cos_scorefusion('score_v_FaceResNet101-JointTraining.txt','score_v_MiniFace-JointTraining.txt')
    #     print("EER: {:.6f}%, minDCF: {:.6f}".format(eer * 100, minDCF))
