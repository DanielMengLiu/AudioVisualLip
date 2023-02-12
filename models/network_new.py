import random
import torch
import torch.nn as nn
import torch.nn.functional  as F
from models.preprocess_new import FeatureAug
from models.feature_new import OnlineFbank
from models.ECAPATDNN_new import ECAPATDNN
from models.MFAConformer_new import MFAConformer
from models.AudioResNet_new import AResNet18, AResNet34, AResNet50, AResNet101, AResNet152, AResNet221, AResNet293
from models.LipMCNN_new import MCNN
from models.CrossModal_new import crossAttentionLayer
from numbers import Number
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
    
class VisualModel(nn.Module):
    def __init__(self, model='MCNN'):
        super(VisualModel, self).__init__()
        if model == 'MCNN':
            self.VisualEncoder = MCNN()
            self.VisualDecoder = ASP_Decoder(hidden_dim=256)

    def forward(self, x_video): 
        x_video = self.VisualEncoder(x_video)
        x_video_ = self.VisualDecoder(x_video)
        return x_video, x_video_

class AudioModel(nn.Module):
    def __init__(self, model='MFA-Conformer', C=1024):
        super(AudioModel, self).__init__()
        self.feature = OnlineFbank()
        self.feataug = FeatureAug() # Spec augmentation
        if model == 'MFA-Conformer':
            self.AudioEncoder = MFAConformer()
            self.AudioDecoder = ASP_Decoder(hidden_dim=256*6)
        elif model == 'ResNet':
            self.AudioEncoder = AResNet50(feat_dim=80, embed_dim=256)
            self.AudioDecoder = SP_Decoder(hidden_dim=20480)
        elif model == 'ECAPA-TDNN':
            self.AudioEncoder = ECAPATDNN(C=C) ##
            self.AudioDecoder = ASP_Decoder(hidden_dim=1536)

    def forward(self, x_audio, aug=False):
        x_audio = x_audio.permute(0, 2, 1)
        x_audio = self.AudioEncoder(x_audio)
        x_audio_ = self.AudioDecoder(x_audio)

        return x_audio, x_audio_

class AudioVisualModel(nn.Module):
    def __init__(self, model='CM', num_blocks=3):
        super(AudioVisualModel, self).__init__()
        self.Affine = Affine(audio_hidden=1536, visual_hidden=256, hidden_dim=128)
        self.CrossModal = CrossModal(num_blocks=num_blocks, hidden_dim=128)
        self.TransAudioDecoder = ASP_Decoder(hidden_dim=128*num_blocks)
        self.TransVisualDecoder = ASP_Decoder(hidden_dim=128*num_blocks)
        
    def forward(self, x_audio, x_video):
        x_audio, x_video = self.Affine(x_audio, x_video)
        x_v2a, x_a2v = self.CrossModal(x_audio, x_video) 
        x_v2a = x_v2a.transpose(1,2)
        x_a2v = x_a2v.transpose(1,2)
        x_v2a_ = self.TransAudioDecoder(x_v2a)
        x_a2v_ = self.TransVisualDecoder(x_a2v)
        return x_v2a_, x_a2v_

class CrossModalDistillationModel(nn.Module):
    def __init__(self, model='CMD', num_blocks=3):
        super(CrossModalDistillationModel, self).__init__()
        self.encoder_a = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 192))
        self.encoder_v = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 192))
        
        self.decoder_a = nn.Sequential(
                nn.Linear(192, 192))
        self.decoder_v = nn.Sequential(
                nn.Linear(192, 192))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x_a, x_v, atar=None, vtar=None, ones=None):
        x_a = x_a.transpose(1,2)
        x_v = x_v.transpose(1,2)
        x_a_ = self.encoder_a(x_a)
        x_v_ = self.encoder_v(x_v)
        # x_a_, x_v_ = F.normalize(x_a_), F.normalize(x_v_)
        
        x_a_ = x_a_ / x_a_.norm(dim=2, keepdim=True)
        x_v_ = x_v_ / x_v_.norm(dim=2, keepdim=True)
        
        # if atar is not None and vtar is not None:
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_a = logit_scale * x_a_ @ x_v_.t()
        logits_per_v = logits_per_a.t()

        return x_a_, x_v_, logits_per_a, logits_per_v
    
# class CrossModalDistillationModel(nn.Module):
#     def __init__(self, model='CMD', num_blocks=3):
#         super(CrossModalDistillationModel, self).__init__()
#         self.encoder_a = nn.Sequential(
#             nn.Linear(192, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 192))
#         self.encoder_v = nn.Sequential(
#             nn.Linear(192, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 192))
        
#         self.decoder_a = nn.Sequential(
#                 nn.Linear(192, 192))
#         self.decoder_v = nn.Sequential(
#                 nn.Linear(192, 192))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
#     def forward(self, x_a, x_v, atar=None, vtar=None, ones=None):
#         x_a_ = self.encoder_a(x_a)
#         x_v_ = self.encoder_v(x_v)
#         # x_a_, x_v_ = F.normalize(x_a_), F.normalize(x_v_)
        
#         x_a_ = x_a_ / x_a_.norm(dim=1, keepdim=True)
#         x_v_ = x_v_ / x_v_.norm(dim=1, keepdim=True)
        
#         # if atar is not None and vtar is not None:
#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_a = logit_scale * x_a_ @ x_v_.t()
#         logits_per_v = logits_per_a.t()

#         return x_a_, x_v_, logits_per_a, logits_per_v

        # else:
        #     kld_a = F.kl_div(F.log_softmax(x_a, dim=1), F.softmax(x_a_, dim=1))
        #     kld_v = F.kl_div(F.log_softmax(x_v, dim=1), F.softmax(x_v_, dim=1))
            
        #     # kld_a = F.cosine_embedding_loss(x_a, x_a_, ones, reduction='mean')
        #     # kld_v = F.cosine_embedding_loss(x_v, x_v_, ones, reduction='mean')
        #     # m_a, lv_a = torch.chunk(x_a_, 2, dim=1)
        #     # m_v, lv_v = torch.chunk(x_v_, 2, dim=1)
        #     # kld_a = 0.5 * (lv_a.exp() + m_a.pow(2) - lv_a - 1).sum(1)
        #     # kld_v = 0.5 * (lv_v.exp() + m_v.pow(2) - lv_v - 1).sum(1)
        #     kld_a = kld_a.mean(0)
        #     kld_v = kld_v.mean(0)
        #     kld = 3 - 0.5 * (kld_a + kld_v)
        #     # samples_a = m_a + torch.rand_like(lv_a) * (0.5 * lv_a).exp()
        #     # samples_v = m_v + torch.rand_like(lv_v) * (0.5 * lv_v).exp()
        #     # predictions_a = self.decoder_a(samples_a)
        #     # predictions_v = self.decoder_v(samples_v)
        #     similar_label = ((atar == vtar) + 0) * 2 - 1
        #     ctl =  F.cosine_embedding_loss(x_a_, x_v_, similar_label, reduction='mean')
        #     return x_a_, x_v_, kld, ctl 
    
class AudioVisualICASSPModel(nn.Module):
    def __init__(self, model='CMR'):
        super(AudioVisualICASSPModel, self).__init__()
        self.CrossModal = CrossModal(num_blocks=1, audio_hidden=1536, visual_hidden=256, hidden_dim=128)
        self.TransAudioDecoder = ASP_Decoder(hidden_dim=128)
        self.TransVisualDecoder = ASP_Decoder(hidden_dim=128)
        self.Fuser = Fuser(hidden_dim=192*2, embed_dim=192)
        
    def forward(self, x_audio, x_video, x_audio_, x_video_):
        x_v2a, x_a2v = self.CrossModal(x_audio, x_video) 
        x_v2a_ = self.TransAudioDecoder(x_v2a)
        x_a2v_ = self.TransVisualDecoder(x_a2v)
        x_ = torch.cat((x_v2a_, x_a2v_), dim=1) #F.normalize(x_audio_),F.normalize(x_video_),
        x_ = self.Fuser(x_)
        return x_
    
class CrossModal(nn.Module):
    def __init__(self, num_blocks=1, hidden_dim=128):
        super(CrossModal, self).__init__()
        self.crossV2A = nn.ModuleList([
            crossAttentionLayer(d_model=hidden_dim, nhead=4) for _ in range(num_blocks)
        ])
        self.crossA2V = nn.ModuleList([
            crossAttentionLayer(d_model=hidden_dim, nhead=4) for _ in range(num_blocks)
        ])
        self.audio_after_norm = torch.nn.LayerNorm(hidden_dim*num_blocks, eps=1e-12) 
        self.video_after_norm = torch.nn.LayerNorm(hidden_dim*num_blocks, eps=1e-12)
              
    def forward(self, x_a, x_v):
        # x_a = x_audio.transpose(1,2)
        # x_v = x_video.transpose(1,2)
        y_a, y_v = x_a, x_v
        out_audio, out_video = [], []
        
        # cross-modal attention
        for V2A in self.crossV2A:
            y_v2a, y_a = V2A(y_v, y_a)
            y_v = y_v2a
            out_audio.append(y_v)
        for A2V in self.crossA2V:
            x_a2v, x_v = A2V(x_a, x_v)
            x_a = x_a2v
            out_video.append(x_a)
        # cat
        y_v2a = torch.cat(out_audio, dim=-1)
        y_v2a = self.audio_after_norm(y_v2a) 
        x_a2v = torch.cat(out_video, dim=-1)
        x_a2v = self.video_after_norm(x_a2v)
        # y_v2a = y_v2a.transpose(1,2)
        # x_a2v = x_a2v.transpose(1,2)
        return y_v2a, x_a2v

class Affine(nn.Module):
    def __init__(self, audio_hidden=1536, visual_hidden=256, hidden_dim=128):
        super(Affine, self).__init__()
        self.affine_A = nn.Linear(audio_hidden, hidden_dim)
        self.affine_V = nn.Linear(visual_hidden, hidden_dim)
              
    def forward(self, x_audio, x_video):
        x_a = x_audio.transpose(1,2)
        x_v = x_video.transpose(1,2)
        x_a = self.affine_A(x_a)
        x_v = self.affine_V(x_v)
        # x_a = x_a.transpose(1,2)
        # x_v = x_v.transpose(1,2)
        return x_a, x_v

# class AudioTemporalDownSample(nn.Module):
#     def __init__(self, dsframe=5):
#         super(AudioTemporalDownSample, self).__init__()

              
#     def forward(self, x_audio, x_video):
#         x_a = x_audio.transpose(1,2)
#         x_v = x_video.transpose(1,2)
#         x_a = self.affine_A(x_a)
#         x_v = self.affine_V(x_v)
#         # x_a = x_a.transpose(1,2)
#         # x_v = x_v.transpose(1,2)
#         return x_a, x_v 
           
class ASP_Decoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ASP_Decoder, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=1), #*3
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, hidden_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.fc1 = nn.Linear(hidden_dim*2, 192)
        self.bn2 = nn.BatchNorm1d(192)
    
    def forward(self, x):
        t = x.size()[-1]
        ## whether global context
        # global_x = torch.cat((x, torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        global_x = x
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4))

        x = torch.cat((mu,sg),1) 
        x = self.bn1(x) 
        x = self.fc1(x)  
        x = self.bn2(x)
        return x

class SP_Decoder(nn.Module):
    def __init__(self, hidden_dim=20480):
        super(SP_Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim*2, 192)
    
    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        x = torch.cat((pooling_mean, pooling_std), 1)
        x = self.fc1(x)
        return x

class AP_Decoder(nn.Module):
    def __init__(self):
        super(AP_Decoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
class Fuser(nn.Module):
    def __init__(self, hidden_dim=192*4, embed_dim=192):
        super(Fuser, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
    
    def forward(self, x):
        x = self.bn1(x) 
        x = self.fc1(x)  
        x = self.bn2(x)
        return x