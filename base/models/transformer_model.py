import torch
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel
from .transformer.Models import Transformer
import Constants

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
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
        parser.add_argument('--tgt_vocab_size', type=int, default=2004)
        parser.add_argument('--proj_share_weight', action='store_true')
        parser.add_argument('--embs_share_weight', action='store_true')
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
        input_shape = input_.shape
        target_shape = target_.shape
        # 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).to(self.device)
        # the _pos variables correspond to the positional encoding (see Transformer paper). These are generated correctly from the collate_fn defined in data/__init__.py
        self.input_pos = input_pos_

        #we collapse all the dimensions of target_ because that is the same way the output of the network is being processed for the cross entropy calculation (see self.forward)
        # here, 0 is the batch dimension, 1 is the window index, 2 is the time dimension, 3 is the output channel dimension
        # self.target = target_.reshape((target_shape[0]*target_shape[1]*target_shape[2]*target_shape[3])).to(self.device)
        # however, we don't collapse all the dimensions of these copies of target, because they need to keep this shape, when fed as inputs to the transformer for training!
        self.target = target_.reshape((target_shape[0]*target_shape[1],target_shape[2]*target_shape[3])).to(self.device)
        self.target_pos = target_pos_

    def forward(self):
        # we are using self.target as mas both for input and target, because we are assuming both sequences are of the same length, for now!
        # if we try for instance, the event based representation of the music transformer, then e would need to change this
        self.output = self.net.forward(self.input.float(),self.target,self.input_pos,self.target,self.target,self.input_pos)

        # using the smoothened loss function from the pytorch Transformer implementation, which also calculates masked accuracy (ignoring the PAD symbol)
        self.loss_ce, n_correct = cal_performance(self.output, self.target[:,1:], smoothing=self.opt.label_smoothing)
        self.metric_accuracy = n_correct/len(self.output)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_ce.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()
