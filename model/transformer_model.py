import math
import logging
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder,TransformerEncoderLayer


class Riiid(nn.Module):
    def __init__(self,dmodel,max_len,nhead=8,nhid=64,nlayers=6,dropout=0.5):
        super(Riiid,self).__init__()

        self.dmodel = dmodel
        self.task_embed = nn.Embedding(num_embeddings=2,embedding_dim=dmodel)
        self.difficulty_embed = nn.Embedding(num_embeddings=12,embedding_dim=dmodel)
        self.tag_embed = nn.Embedding(num_embeddings=188,embedding_dim=dmodel)
        self.elapsetime_embed = nn.Embedding(num_embeddings=301,embedding_dim=dmodel)
        self.part_embed = nn.Embedding(num_embeddings=8,embedding_dim=dmodel)
        # self.pos_encoder = PositionalEncoding(dmodel)
        self.pos_embed = nn.Embedding(max_len, dmodel)
        
        encoder_layer = TransformerEncoderLayer(d_model=dmodel,
            nhead=nhead,dim_feedforward=nhid,dropout=dropout)

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
            num_layers=nlayers)

        self.fc = nn.Linear(dmodel,2)
        
        self.init_weights()


    def init_weights(self):
        initrange = 0.1

        self.task_embed.weight.data.uniform_(-initrange, initrange)
        self.difficulty_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)
        self.elapsetime_embed.weight.data.uniform_(-initrange, initrange)
        self.part_embed.weight.data.uniform_(-initrange, initrange)
        self.pos_embed.weight.data.uniform_(-initrange, initrange)

        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)


    def forward(self,x_pos,x_task,x_diff,x_tag,x_et,x_part,pad_mask):
        
        embeds = self.task_embed(x_task) + \
            self.difficulty_embed(x_diff) + \
            self.tag_embed(x_tag) + \
            self.elapsetime_embed(x_et) + \
            self.part_embed(x_part) + \
            self.pos_embed(x_pos)
        #TODO difficulty_embed
        # print(torch.min(embeds),torch.max(embeds))
        embeds = embeds.transpose(0,1)
        embeds = embeds * math.sqrt(self.dmodel)

        output = self.encoder(src=embeds,src_key_padding_mask=pad_mask)
        output = output.transpose(1,0)

        output = self.fc(output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)