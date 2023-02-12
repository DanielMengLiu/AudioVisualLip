import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.convolution import ConvolutionModule


class crossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(crossAttentionLayer, self).__init__()
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ffn11 = PositionwiseFeedForward(d_model, d_model//4, 0.1)
        self.ffn12 = PositionwiseFeedForward(d_model, d_model//4, 0.1)
        self.ffn2 = PositionwiseFeedForward(d_model, d_model//4, 0.1)
        self.conv = ConvolutionModule(d_model)
        self.norm_ff11 = nn.LayerNorm(d_model, eps=1e-12)  # for the FNN module
        self.norm_ff12 = nn.LayerNorm(d_model, eps=1e-12)  # for the FNN module
        self.norm_ff2 = nn.LayerNorm(d_model, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model, eps=1e-12)  # for the MHA module
        self.norm_conv = nn.LayerNorm(d_model, eps=1e-12)  # for the CNN module
        # self.norm_final = torch.nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        # self.tar_cls_token = nn.Parameter(torch.randn(1, 1, 256))
        # self.src_cls_token = nn.Parameter(torch.randn(1, 1, 256))
        
    def forward(self, target, source): #a->v
        # target = torch.cat((self.tar_cls_token.repeat(target.shape[0],1,1), target), dim=-2)  ########
        # source = torch.cat((self.src_cls_token.repeat(target.shape[0],1,1), source), dim=-2)  ##########
        residual = target
        target_ = residual + self.dropout(self.ffn11(target))
        target_ = self.norm_ff11(target_)

        residual = source
        source_ = residual + self.dropout(self.ffn12(source))
        source_ = self.norm_ff12(source_)

        t2s = self.dropout(self.cross_attn(source_, target_, target_)[0])
        t2s_ = self.norm_mha(t2s)
        new = torch.maximum(t2s_, source_) 
        # new = t2s_ + source_  # ablation
        
        residual = new
        new = residual + self.dropout(self.conv(new)[0])
        new_ = self.norm_conv(new)

        residual = new_
        final = residual + self.dropout(self.ffn2(new_))
        final_ = self.norm_ff2(final)

        # cls_token = final_[:,0,:]
        return final_, source_  # target, source