import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel
from .transformer.Models import Transformer
import Constants
from process_scripts.data_processing.state_space_functions import get_block_sequence_with_deltas
from transformer.Translator import Translator

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    # gold = gold.contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    # print(pred.shape, gold.shape)
    # print(pred, gold)
    # name = input("Enter your name: ")   # Python 3
    non_pad_mask = gold.ne(Constants.PAD_STATE)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD_STATE)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='mean')

    return loss



class TransformerModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.loss_names = ['ce']
        self.metric_names = ['accuracy']
        self.module_names = ['']  # changed from 'model_names'
        self.schedulers = []
        self.net = Transformer(
                d_tgt=opt.d_tgt,
                d_src=opt.d_src,
                n_tgt_vocab=opt.tgt_vocab_size,
                n_src_vocab=0,
                len_max_seq=opt.max_token_seq_len,
                src_vector_input=opt.src_vector_input,
                tgt_vector_input=opt.tgt_vector_input,
                tgt_emb_prj_weight_sharing=opt.proj_share_weight,
                emb_src_tgt_weight_sharing=opt.embs_share_weight,
                d_k=opt.d_k,
                d_v=opt.d_v,
                d_model=opt.d_model,
                d_word_vec=opt.d_word_vec,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                n_head=opt.n_head,
                dropout=opt.dropout)

        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_ce = None

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hidden_dim', type=int, default=100)
        parser.add_argument('--d_src', type=int, default=24)
        parser.add_argument('--d_tgt', type=int, default=24)
        parser.add_argument('--tgt_vocab_size', type=int, default=2004)
        parser.add_argument('--proj_share_weight', action='store_true')
        parser.add_argument('--embs_share_weight', action='store_true')
        parser.add_argument('--src_vector_input', action='store_true')
        parser.add_argument('--tgt_vector_input', action='store_true')
        parser.add_argument('--label_smoothing', action='store_true')
        parser.add_argument('--d_k', type=int, default=64)
        parser.add_argument('--d_v', type=int, default=64)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--d_word_vec', type=int, default=512)
        parser.add_argument('--d_inner_hid', type=int, default=2048)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--n_head', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_,input_pos_ = data['input']
        target_,target_pos_ = data['target']
        ### THING BELOG IS IF WE FEED VECTORS AS INPUT TO THE DECODER!
        if self.opt.tgt_vector_input:
            input_block_deltas_ = target_[:,:,1:,:]
            target_block_sequence_ = target_[:,:,:1,:]
            input_block_deltas_shape = input_block_deltas_.shape
        else:
            target_block_sequence_ = target_

        target_block_sequence_shape = target_block_sequence_.shape
        input_shape = input_.shape
        input_pos_shape = input_pos_.shape
        # 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(0,2,1).to(self.device)
        print(self.input.shape)
        # self.input_pos = input_pos_.reshape((input_pos_shape[0], input_pos_shape[1])).to(self.device)
        self.input_pos = input_pos_
        #we permute the dimensions because transformer input expects (batch_size, time, input_dim)
        # the _pos variables correspond to the positional encoding (see Transformer paper). These are generated correctly from the collate_fn defined in data/__init__.py

        # here, 0 is the batch dimension, 1 is the window index, 2 is the output channel dimension, 3 is the time dimension
        ### THIS IS WE FEED VECTORS AS INPUT TO THE DECODER!
        if self.opt.tgt_vector_input:
            self.input_block_deltas = input_block_deltas_.reshape((input_block_deltas_shape[0]*input_block_deltas_shape[1],input_block_deltas_shape[2],input_block_deltas_shape[3])).permute(0,2,1).to(self.device)
        #we permute the dimensions because transformer input expects (batch_size, time, input_dim)

        self.target_block_sequence = target_block_sequence_.reshape((target_block_sequence_shape[0]*target_block_sequence_shape[1],target_block_sequence_shape[2]*target_block_sequence_shape[3])).to(self.device)
        # in other models, we collapse all the dimensions of target_ because that is the same way the output of the network is being processed for the cross entropy calculation (see self.forward)
        # however, we don't collapse all the dimensions of these copies of target, because they need to keep this shape, when fed as inputs to the transformer for training!

        # self.target_pos = target_pos_

    def forward(self):
        # we are using self.target as mas both for input and target, because we are assuming both sequences are of the same length, for now!
        # if we try for instance, the event based representation of the music transformer, then e would need to change this
        if self.opt.tgt_vector_input:
            self.output = self.net.forward(self.input.float(),self.target_block_sequence,self.input_pos,self.input_block_deltas.float(), self.target_block_sequence,self.target_block_sequence,self.input_pos)
        else:
            self.output = self.net.forward(self.input.float(),self.target_block_sequence,self.input_pos,self.target_block_sequence, self.target_block_sequence,self.target_block_sequence,self.input_pos)

        # using the smoothened loss function from the pytorch Transformer implementation, which also calculates masked accuracy (ignoring the PAD symbol)
        self.loss_ce, n_correct = cal_performance(self.output, self.target_block_sequence[:,1:], smoothing=self.opt.label_smoothing)
        self.metric_accuracy = n_correct/len(self.output)

    def generate(self, features, json_file, bpm, unique_states, temperature, use_beam_search=False, generate_full_song=False):
        opt = self.opt

        y = features
        y = np.concatenate((np.zeros((y.shape[0],1)),y),1)
        y = np.concatenate((y,np.zeros((y.shape[0],1))),1)
        if opt.using_bpm_time_division:
            beat_duration = 60/bpm #beat duration in seconds
            beat_subdivision = opt.beat_subdivision
            sample_duration = beat_duration * 1/beat_subdivision #sample_duration in seconds
        else:
            sample_duration = opt.step_size
        sequence_length_samples = y.shape[1]
        sequence_length = sequence_length_samples*sample_duration

        ## BLOCKS TENSORS ##
        one_hot_states, states, state_times, delta_forward, delta_backward, indices = get_block_sequence_with_deltas(json_file,sequence_length,bpm,sample_duration,top_k=2000,states=unique_states,one_hot=True,return_state_times=True)
        if not generate_full_song:
            truncated_sequence_length = min(len(states),opt.max_token_seq_len)
        else:
            truncated_sequence_length = len(states)
        indices = indices[:truncated_sequence_length]
        delta_forward = delta_forward[:,:truncated_sequence_length]
        delta_backward = delta_backward[:,:truncated_sequence_length]

        input_forward_deltas = torch.tensor(delta_forward).unsqueeze(0).long()
        input_backward_deltas = torch.tensor(delta_backward).unsqueeze(0).long()
        if opt.tgt_vector_input:
            input_block_sequence = torch.tensor(one_hot_states).unsqueeze(0).long()
            input_block_deltas = torch.cat([input_block_sequence,input_forward_deltas,input_backward_deltas],1)

        y = y[:,indices]
        input_windows = [y]
        song_sequence = torch.tensor(input_windows)
        song_sequence = (song_sequence - song_sequence.mean())/torch.abs(song_sequence).max().float()
        if not opt.tgt_vector_input:
            song_sequence = torch.cat([song_sequence,input_forward_deltas.double(),input_backward_deltas.double()],1)

        src_pos = torch.tensor(np.arange(len(indices))).unsqueeze(0)
        src_mask = torch.tensor(Constants.NUM_SPECIAL_STATES*np.ones(len(indices))).unsqueeze(0)

        ## actually generate level ##
        translator = Translator(opt,self)
        # need to pass to beam .advance, the length of sequence :P ... I think it makes sense
        if opt.tgt_vector_input:
            raise NotImplementedError("Need to implement beam search for Transformer target vector inputs (when we attach deltas to target sequence)")
        else:
            if use_beam_search:
                all_hyp, all_scores = translator.translate_batch(song_sequence.permute(0,2,1).float(), src_pos, src_mask,truncated_sequence_length)
                generated_sequence = all_hyp[0][0]
            else:
                generated_sequence = translator.sample_translation(song_sequence.permute(0,2,1).float(), src_pos, src_mask,truncated_sequence_length, temperature)
        # return state_times, all_hyp[0] # we are for now only supporting single batch generation..
        return state_times, generated_sequence


    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_ce.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()
