
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
from models.loss import AAMsoftmax
from models.network_new import AudioModel
from utils.eval_metrics_new import *


################# GLOBAL CONFIGURATION ##############################
MODE  = 'train'           # train | test | finetune
#####################################################################

SEED  = 1022              # fixed random seed for fair comparison
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings('ignore')

with open('./conf/savedsolution_audio_vox_IT.yaml', 'r') as f:
    OPTS = yaml.load(f, Loader=CLoader)
# shutil.copyfile('./conf/config_audio_lrs3_new.yaml', './conf/savedsolution_audio_lrs3_IT.yaml')

# Basic functions for train and test phase
def get_model(model):
    return AudioModel(model, C=1024)

class Trainer(object):
    def __init__(self):
        self.stageopts = OPTS[MODE]
        self.modelname = self.stageopts['model']
        self.modelopts = OPTS[self.modelname]
        self.resume = self.stageopts['resume']
        self.audioemb_dim = self.modelopts['embedding_dim']

        # neural network
        audiomodel = get_model(self.modelname)
        print('audio model parameters_count: %.2fM' % (sum(p.numel() for p in audiomodel.parameters() if p.requires_grad)/1e6))

        device_ids, device_num = self.stageopts['gpus'], len(self.stageopts['gpus'])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in device_ids])
        self.device_ids = [x for x in range(len(device_ids))]
        self.device = torch.device('cuda') # : +str(device_ids[0])
        self.audiomodel = torch.nn.DataParallel(audiomodel.to(self.device), device_ids=self.device_ids)

        self.featureopts = OPTS[self.modelopts['feature']]
        self.dataopts = {}
        # self.dataopts = {**{'seconds':self.stageopts['seconds']}, **{'sample_rate':self.featureopts['sample_rate']}}
        self.dataopts = {'seconds':self.stageopts['seconds'], 'sample_rate':self.featureopts['sample_rate']}
        for aug in self.stageopts['augmentation'].keys():
            # self.dataopts = {**self.dataopts, **{aug:self.stageopts['augmentation'][aug]}}
            self.dataopts[aug] = self.stageopts['augmentation'][aug]
        for traindata in ['train_manifest', 'train_audiodir', 'musan_path', 'rir_path']: # for train and aug
            # self.dataopts = {**self.dataopts, **{traindata:OPTS[traindata]}}
            self.dataopts[traindata] = OPTS[traindata]
        for testdata in ['test_trial', 'test_audiodir']: # for val
            # self.dataopts = {**self.dataopts, **{testdata:OPTS['test_vox1'][testdata]}}
            self.dataopts[testdata] = OPTS['test_vox1'][testdata]

        self.trainset = datasets.AudioTrainset(self.dataopts)
        self.trainloader = DataLoader(self.trainset, shuffle=True, batch_size=self.stageopts['batchsize'], num_workers=4*device_num, drop_last=True)

        times = len(self.stageopts['augmentation']['speedperturb']) # whether do speaker augmentation by speed perturb
        self.audiocriterion = AAMsoftmax(n_class=self.trainset.n_spk*times, m=self.stageopts['margins'][1], s=self.stageopts['scale'], em_dim=self.audioemb_dim).to(self.device) 

        param_groups = [{'params': self.audiomodel.parameters()}, 
                        {'params': self.audiocriterion.parameters()},
                        ]
 
        if self.stageopts['optimizer'] == 'sgd':
            self.optimopts = OPTS['sgd']
            self.optim = optim.SGD(param_groups, self.optimopts['init_lr'], nesterov=self.optimopts['nesterov'], momentum = self.optimopts['momentum'], weight_decay = self.optimopts['weight_decay'])
        elif self.stageopts['optimizer'] == 'adam':
            self.optimopts = OPTS['adam']
            self.optim = optim.Adam(param_groups, lr=self.optimopts['init_lr'] , weight_decay=self.optimopts['weight_decay'])
        if self.stageopts['lr_scheduler'] == 'steplr':
            # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=2, gamma=0.8)
            # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=0.97)
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10,15], gamma=0.1)
        elif self.stageopts['lr_scheduler'] == 'cycliclr':
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optim, cycle_momentum=False, base_lr=0.000001, max_lr=0.001,step_size_up=2000,step_size_down=2000)
        
        self.eers = []
        self.dcfs = []
        self.current_epoch, self.epochs = 0, self.stageopts['epochs']
        if self.resume == 'None':
            # sp = 'sp'+str(len(self.stageopts['augmentation']['speedperturb'])) 
            self.exp = '_'.join(['vox', 'independent-training', self.modelname])
        else:
            self.exp = self.resume.split('/')[1]

        # continue training
        if self.resume != 'None' and os.path.exists(self.resume):
            self._load()
        else:
            print('Train from scratch: there is no specified resume.')
        
        self.wandb = self.stageopts['wandb']
        if self.wandb != False:
            config = {
                "learning_rate": self.optimopts['init_lr'],
                "epochs": self.epochs,
                "batch_size": self.stageopts['batchsize']
            }
            wandb.init(project="Short-Short-New", config=config)


    def _load(self):
        print('loading model from {}'.format(self.resume))

        if not os.path.exists(self.resume):
            print('No pretrained model exists!')
        ckpt = torch.load(self.resume)
        if 'audiostate_dict' in ckpt.keys():
            self.audiomodel.load_state_dict(ckpt['audiostate_dict'])
        # self.current_epoch = ckpt['epoch']
        # if 'optimizer' in ckpt.keys():
        #     self.optim.load_state_dict(ckpt['optimizer'])
        if 'audiocriterion' in ckpt.keys():
            self.audiocriterion = ckpt['audiocriterion']
                    
    def _train(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            self._train_epoch()
            self.lr_scheduler.step()

    def _train_epoch(self):
        self.audiomodel.train()
        self.audiocriterion.train()
        audiosum_loss, audiosum_samples, audiocorrect = 0, 0, 0
        progress_bar = tqdm(self.trainloader)
                
        for batch_idx, (audiofeats, targets_label) in enumerate(progress_bar):
            self.optim.zero_grad()
            audiofeats = audiofeats.to(self.device)
            targets_label = targets_label.to(self.device)
            _, output_audio_ = self.audiomodel(audiofeats, aug=True)
            
            audioloss, audiologits = self.audiocriterion(output_audio_, targets_label)
            audioloss.backward()

            audiosum_samples += len(audiofeats)
            _, audioprediction = torch.max(audiologits, dim = 1)
            audiocorrect += (audioprediction == targets_label).sum().item()

            # linear warm up for 2000 steps
            # if self.current_epoch == 1 and batch_idx < 2000:
            #     lr_scale = min(1., float(batch_idx + 1) / float(2000))
            #     for _, pg in enumerate(self.optim.param_groups):
            #         pg['lr'] = lr_scale * 0.001

            self.optim.step()
         
            audiosum_loss += audioloss.item() * len(targets_label)
            progress_bar.set_description(
                    'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] audioLoss: {:.4f} audioAcc: {:.4f}%' #  
                    .format(self.current_epoch, batch_idx + 1,
                    len(self.trainloader), 100. * (batch_idx + 1) / len(self.trainloader),
                    audiosum_loss / audiosum_samples, 100. * audiocorrect / audiosum_samples))
            if self.wandb != False:
                wandb.log({"audioLoss":audiosum_loss/audiosum_samples,
                        "lr": self.optim.state_dict()['param_groups'][0]['lr'],
                        "audioAcc":100.*audiocorrect/audiosum_samples})
        self._save('exp/{}/net_{}.pth'.format(self.exp, self.current_epoch))
        # flexible log
        val_step = 1
        if self.current_epoch % val_step == 0:
            eer, minDCF = self._eval_network(stage='val', num_workers=1)
            self.eers.append(eer)         
            self.dcfs.append(minDCF)
            if not os.path.isdir('log/'):
                os.mkdir('log')
            with open('log/'+self.exp+'-training.log', "a+") as score_file:   
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, minDCF %2.4f, bestEER %2.2f%%, bestminDCF %2.4f\n"  \
                                %(self.current_epoch, self.optim.state_dict()['param_groups'][0]['lr'], audiosum_loss / audiosum_samples, \
                                100. * audiocorrect / audiosum_samples, self.eers[-1], self.dcfs[-1], min(self.eers), min(self.dcfs)))
                score_file.flush()
        return audiosum_loss / audiosum_samples
        
    def _save(self, modelpath):
        torch.save({'epoch': self.current_epoch,
                    'audiostate_dict': self.audiomodel.state_dict(),
                    'audiocriterion': self.audiocriterion, 
                    'optimizer': self.optim.state_dict()},
                    modelpath) 

    def _extract_embedding(self, stage, filelist):  # stage='val'
        testset = datasets.AudioTestset(self.dataopts, filelist, stage='val')
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

        localmodel = self.audiomodel.module
        localmodel = localmodel.to(self.device)
        localmodel.eval()
        # print('Extracting test embeddings for {}: '.format(dataset))
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        os.makedirs(emb_dir, exist_ok = True)
        with torch.no_grad():
            for signal_global, signal_local, utt in tqdm(testloader):
                utt = utt[0]
                spk_dir = os.path.join(emb_dir, os.path.dirname(utt))
                os.makedirs(spk_dir, exist_ok=True)
                signal_global = signal_global.squeeze(0).to(self.device)
                signal_local = signal_local.squeeze(0).to(self.device)
                _, embedding_global = localmodel(signal_global, aug=False)
                _, embedding_local = localmodel(signal_local, aug=False)
                embedding_global = embedding_global.cpu().numpy()
                embedding_local = embedding_local.cpu().numpy()
                np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt.split('.')[0])), [embedding_global, embedding_local])
        del localmodel
        torch.cuda.empty_cache()

    def _score_embedding(self, stage, trials, score_dict):  
        # GRID scoring for global and local embeddings and save score in score/MODELDIR/score_TRIAL.txt
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        for line in trials:
            embedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
            embedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
            embedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
            embedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
            embedding_11, embedding_12, embedding_21, embedding_22 = F.normalize(embedding_11),F.normalize(embedding_12),F.normalize(embedding_21),F.normalize(embedding_22)
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            score_dict[line] = [score, int(line.split()[0])] # score, label

    def _eval_network(self, stage, num_workers=1):
        # prepare for multiprocessing of embedding extraction
        files, utts = [], []
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

        self.audiomodel.cpu()
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

        num_workers = 40
        score_dict = Manager().dict()
        part = list(range(0, len(lines)+1, int(len(lines)//num_workers)))
        part[-1] = len(lines)
        trials = [lines[part[i]:part[i+1]] for i in range(num_workers)]
        args = [(stage, trials[i], score_dict) for i in range(num_workers)]
        jobs = [Process(target=self._score_embedding, args=(a)) for a in args]
        for j in jobs: j.start()
        for j in jobs: j.join()
        
        scores, labels = [], []
        for line in lines:
            scores.append(score_dict[line][0])
            labels.append(score_dict[line][1])

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("EER: {:.6f}%, minDCF: {:.6f}".format(EER, minDCF))
        return EER, minDCF

    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)
 
    def __call__(self):
        print("[Model is saved in: {}]".format(self.exp))
        # print("Data opts: ")
        # self._print_config(self.dataopts)
        # print("Model opts: ")
        # self._print_config(self.modelopts)
        # print("Stage opts: ")
        # self._print_config(self.stageopts)
        os.makedirs('exp/{}'.format(self.exp), exist_ok = True)
        self._train()


class Tester(object):
    def __init__(self):
        self.stageopts = OPTS[MODE]
        self.modelname = self.stageopts['model']
        self.modelopts = OPTS[self.modelname]
        self.audioemb_dim = self.modelopts['embedding_dim']
        self.resume = self.stageopts['resume']
        self.exp = self.resume.split('/')[1]

        self.gpus = self.stageopts['gpus']
        self.device_ids = [x for x in range(len(self.gpus))]

        self.featureopts = OPTS[self.modelopts['feature']]
        self.dataopts = {}
        for data in ['train_manifest', 'train_audiodir', 'cohort_manifest']: # for submean, cohort, and test
            # self.dataopts = {**self.dataopts, **{data:OPTS[data]}}
            self.dataopts[data] = OPTS[data]
        for testdata in ['test_trial', 'test_audiodir',]: # for test
            # self.dataopts = {**self.dataopts, **{testdata:OPTS[self.stageopts['data']][testdata]}}
            self.dataopts[testdata] = OPTS[self.stageopts['data']][testdata]

    def _extract_submeanembedding(self, stage):  
        self.submeanset = datasets.AudioSubmeanset(self.dataopts)
        self.submeanloader = DataLoader(self.submeanset, batch_size = 400, shuffle=False, num_workers=4, drop_last=True)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.gpus])
        device = torch.device('cuda')
        testmodel = get_model(self.modelname)
        testmodel = torch.nn.DataParallel(testmodel.to(device), device_ids=[x for x in range(len(self.gpus))])
        ckpt = torch.load(self.resume)
        if 'audiostate_dict' in ckpt.keys():
            testmodel.load_state_dict(ckpt['audiostate_dict'], strict=False)

        testmodel.eval()
        print('Extracting submean embedding: ')
        os.makedirs('exp/{}/{}_emb/'.format(self.exp, stage), exist_ok = True)
        sum_output_audio = None
        with torch.no_grad():
            dataloader = self.submeanloader
            for audiofeat, utt in tqdm(dataloader):
                audiofeat = audiofeat.to(device)
                utt = utt[0]
                _, output_audio_ = testmodel(audiofeat, aug=False) 
                output_audio_ = output_audio_.cpu().numpy()
                if sum_output_audio is None:
                   sum_output_audio =  np.mean(output_audio_, axis=0)
                else:
                   sum_output_audio += np.mean(output_audio_, axis=0)
        avg_output_audio = sum_output_audio / len(dataloader)
        np.save(os.path.join('exp/{}/{}_emb/'.format(self.exp, stage),'submean.npy'), avg_output_audio)

    def _extract_embedding(self, stage, filelist, gpu):  # test | cohort | submean 
        testset = datasets.AudioTestset(self.dataopts, filelist, stage)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device('cuda')
        testmodel = get_model(self.modelname)
        testmodel = torch.nn.DataParallel(testmodel.to(device), device_ids=[0])
        ckpt = torch.load(self.resume)
        if 'audiostate_dict' in ckpt.keys():
            testmodel.load_state_dict(ckpt['audiostate_dict'], strict=False)

        testmodel.module.eval()
        # print('Extracting test embeddings for {}: '.format(dataset))
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        os.makedirs(emb_dir, exist_ok = True)
        with torch.no_grad():
            for signal_global, signal_local, utt in tqdm(testloader):
                utt = utt[0]
                spk_dir = os.path.join(emb_dir, os.path.dirname(utt))
                os.makedirs(spk_dir, exist_ok=True)

                signal_global = signal_global.squeeze(0).to(device)
                signal_local = signal_local.squeeze(0).to(device)

                _, embedding_global = testmodel(signal_global, aug=False)
                _, embedding_local = testmodel(signal_local, aug=False)

                embedding_global = embedding_global.cpu().numpy()
                embedding_local = embedding_local.cpu().numpy()
                np.savez_compressed(os.path.join(spk_dir, os.path.basename(utt.split('.')[0])), [embedding_global, embedding_local])

    def _score_embedding(self, stage, trials, score_dict):  
        emb_dir = os.path.join('exp/{}/{}_emb'.format(self.exp, stage))
        for line in trials:
            embedding_11 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
            embedding_12 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-2].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
            embedding_21 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][0])
            embedding_22 = torch.FloatTensor(np.load(os.path.join(emb_dir, line.split()[-1].split('.')[0]+'.npz'), allow_pickle=True)['arr_0'][1])
            embedding_11, embedding_12, embedding_21, embedding_22 = F.normalize(embedding_11),F.normalize(embedding_12),F.normalize(embedding_21),F.normalize(embedding_22)
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

        print('loading model from {}'.format(self.resume))
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

        num_workers = 20
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

    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)


if __name__ == '__main__':
    if MODE == 'train' or MODE == 'finetune':
        trainer = Trainer()
        trainer()
    # voxceleb dataset test
    elif MODE == 'test':
        stage = 'test'   # test | cohort | submean
        tester = Tester()
        if stage in ['test', 'cohort']:
            tester._eval_network(stage=stage, num_workers=24)
        if stage in ['submean']:
            tester._extract_submeanembedding(stage=stage)
    # elif MODE == 'extract_submean_embedding':
    #     trainer.extract_submeanembedding()    
    # elif MODE == 'featurefusion':
    #     eer, minDCF = utils.eer_cos_featurefusion(trainer.log_time, dataset='lrs3', trial=trainer.data_opts['test_lrs3']['test_trial_lrs3'])
    #     print("EER: {:.6f}%, minDCF: {:.6f}".format(eer * 100, minDCF))
    # elif MODE == 'scorefusion':
    #     eer, minDCF = utils.eer_cos_scorefusion('score_v_FaceResNet101-JointTraining.txt','score_v_MiniFace-JointTraining.txt')
    #     print("EER: {:.6f}%, minDCF: {:.6f}".format(eer * 100, minDCF))
