import sys
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber")
sys.path.append("/home/guillefix/code/beatsaber/base/models")
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
import json
import librosa
import torch
import pickle
import os

from stateSpaceFunctions import feature_extraction_hybrid_raw

#%%
experiment_name = "lstm_testing/"

opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
opt["data_dir"] = "../AugDataTest"
opt["dataset_name"] = "general_beat_saber"
opt["time_shifts"] = 1
opt["concat_outputs"] = False
opt["using_sync_features"] = True
opt["reduced_state"] = True
opt["d_src"] = 24
opt["tgt_vocab_size"] = 2001
opt["max_token_seq_len"] = 500
opt["proj_share_weight"] = True
opt["embs_share_weight"] = False
opt["d_k"] = 64
opt["d_v"] = 64
opt["d_model"] = 512
opt["d_word_vec"] = 512
opt["d_inner_hid"] = 2048
opt["n_layers"] = 4
opt["n_head"] = 8
opt["dropout"] = 0.1
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

if opt.model=='wavenet' or opt.model=='adv_wavenet':
    if not opt.gpu_ids:
        receptive_field = model.net.receptive_field
    else:
        receptive_field = model.net.module.receptive_field
else:
    receptive_field = 1


train_dataset = create_dataset(opt, receptive_field=receptive_field)
train_dataset.setup()
train_dataloader = create_dataloader(train_dataset)
#%%
batch = next(iter(train_dataloader))

batch.keys()
# batch['input'][0]
input_ = batch['input']
target_ = batch['target']
input_shape = input_.shape
target_shape = target_.shape
# 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).to("cuda")
target = target_.reshape((target_shape[0]*target_shape[1],target_shape[2]*target_shape[3])).to("cuda")
target = torch.cat([torch.zeros(target.shape[0],1).long().cuda(),target],1)
input = torch.cat([torch.zeros(input.shape[0],input.shape[1],1).cuda(),input],2)

import imp; import transformer;imp.reload(transformer.Models)
from transformer.Models import Transformer


net = Transformer(
    opt.d_src,
    opt.tgt_vocab_size,
    opt.max_token_seq_len,
    tgt_emb_prj_weight_sharing=opt.proj_share_weight,
    emb_src_tgt_weight_sharing=opt.embs_share_weight,
    d_k=opt.d_k,
    d_v=opt.d_v,
    d_model=opt.d_model,
    d_word_vec=opt.d_word_vec,
    d_inner=opt.d_inner_hid,
    n_layers=opt.n_layers,
    n_head=opt.n_head,
    dropout=opt.dropout).to("cuda")

# input /=-1000
src_mask = torch.ones(input.shape[0],input.shape[2])
src_pos = torch.Tensor(range(input.shape[-1])).long()
src_pos = src_pos.repeat(input.shape[0],1)
input = input.cuda()
src_mask = src_mask.cuda()
src_pos = src_pos.cuda()
target = target.cuda()
# input
# target
output = net(input,src_mask,src_pos,target,src_mask,src_pos)

# net.encoder.src_word_emb(input.permute(0,2,1))
# enc_output, *_ = net.encoder(input, src_mask, src_pos)

# dec_output, *_ = net.decoder(target, src_pos, src_mask, src_mask, enc_output)

output.shape

import matplotlib.pyplot as plt
thing = output[3].detach().cpu().numpy()
thing.max()
plt.plot(thing)
