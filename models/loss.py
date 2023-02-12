'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys 
sys.path.append("..") 

def distillation_loss(student_scores, teacher_scores, T, reduction_kd='mean', reduction_nll='mean'):
    d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(student_scores / T, dim=1),
                                                    F.softmax(teacher_scores / T, dim=1)) * T * T
    return d_loss

def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    # if normalized_patience:
    #     teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
    #     student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()
    
class CosineEmbeddingLoss(nn.Module):
    def __init__(self):
        super(CosineEmbeddingLoss, self).__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.cel = nn.CosineEmbeddingLoss()
        
    def forward(self, embedding1, embedding2, similarlabel):
        loss = self.cel(embedding1, embedding2, similarlabel)
        # Uncertainty-based weighting
        squared_sigma = torch.exp(self.log_sigma)**2 
        loss = loss / squared_sigma + squared_sigma
        #print(squared_sigma)
        return loss
    
class CrossEntropy(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CrossEntropy, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_size, num_classes)
        
    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits + 1e-8, labels)
        return loss, logits
 
class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s, em_dim):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, em_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        #Uncertainty-based weighting
        # self.log_sigma = nn.Parameter(torch.zeros(1, device='cuda'))
        
    def forward(self, x, label=None):    
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        # prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0][0]
        # Uncertainty-based weighting
        # squared_sigma = torch.exp(self.log_sigma)**2 
        # loss = loss / squared_sigma + squared_sigma
        #print(squared_sigma)
        return loss, output

def ValueLoss(ace, vce, akl, vkl, valuerate, wce=0.4, wkl=0.1): # cross-entropy, KL divergence loss
    # valueteachercnt = round(len(ace) * valuerate)
    # valuestudentcnt = round(valueteachercnt * valuerate)
    # # ace, vce, akl, vkl = ace.cpu(), vce.cpu(), akl.cpu(), vkl.cpu()
    # sortedindex_vce = torch.argsort(vce, descending=False)[:valueteachercnt].tolist()  # small vce
    # ace_1 = ace[sortedindex_vce]
    # vce_1 = vce[sortedindex_vce]
    # akl_1 = akl[sortedindex_vce]
    # vkl_1 = vkl[sortedindex_vce]
    # sortedindex_ace = torch.argsort(ace_1, descending=True)[:valuestudentcnt].tolist()  # big ace
    # ace_2 = ace_1[sortedindex_ace]
    # vce_2 = vce_1[sortedindex_ace]
    # akl_2 = akl_1[sortedindex_ace]
    # vkl_2 = vkl_1[sortedindex_ace]
    # avloss = torch.mean(wce*ace_2 + wce*vce_2 + wkl*akl_2 + wkl*vkl_2)

    valueteachercnt = round(len(ace) * valuerate)
    avloss = 0.9*ace + 0.9*vce + 0.1*akl + 0.1*vkl
    sortedindex_avloss = torch.argsort(avloss, descending=False)[:valueteachercnt].tolist()  # small vce
    ace_2 = ace[sortedindex_avloss]
    vce_2 = vce[sortedindex_avloss]
    akl_2 = akl[sortedindex_avloss]
    vkl_2 = vkl[sortedindex_avloss]
    avloss = avloss[sortedindex_avloss]

    return torch.mean(avloss), torch.mean(ace_2), torch.mean(vce_2), torch.mean(akl_2), torch.mean(vkl_2)

def CoregularizationLoss(aloss, vloss, traloss, trvloss, traalignloss, trvalignloss): # 6 loss
    avloss = 0.2*torch.mean(aloss) + 0.2*torch.mean(vloss) + 0.2*torch.mean(traloss) + 0.2*torch.mean(trvloss) + 0.1*torch.mean(traalignloss) + 0.1*torch.mean(trvalignloss)
    return torch.mean(avloss), torch.mean(aloss), torch.mean(vloss), torch.mean(traloss), torch.mean(trvloss), torch.mean(traalignloss), torch.mean(trvalignloss)