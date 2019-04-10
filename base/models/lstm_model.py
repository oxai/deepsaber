import torch
import torch.nn.functional as F
import torch.nn as nn


class LSTMNet(nn.Module):

    def __init__(self, opt):
        super().__init__()
        '''  hidden_dim: LSTM Output Dimensionality
             embedding_dim: LSTM Input Dimensionality, which 
             is 12 for chromagram and 20 (default) for MFCC
        '''

        self.hidden_dim = opt.hidden_dim
        if opt.embedding_dim is None:
            embedding_dim = 12  # Chroma, can be changed for MFCC
        else:
            embedding_dim = opt.embedding_dim

        self.lstm = nn.LSTM(embedding_dim, opt.hidden_dim)  # Define the LSTM
        self.hidden_to_state = nn.Linear(opt.hidden_dim,
                                         opt.vocab_size)  # vocab_size used so far is 2001 by default (2000 + empty state)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)  # Input is a 3D Tensor: [length, batch_size, dim] !! Feeder Functions Needed
        state_preoutput = self.hidden_to_state(lstm_out)  # lstm_out shape compatibility (Need to transpose?)
        state_output = F.log_softmax(state_output, dim=1)
        return state_output