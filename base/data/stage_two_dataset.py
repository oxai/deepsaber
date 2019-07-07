from pathlib import Path
from itertools import tee
import numpy as np
import torch
import librosa
from base.data.base_dataset import BaseDataset
import json
from math import floor, ceil
import pickle
unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
# feature_name = "chroma"
# feature_size = 24
# number_reduced_states = 2000
from .level_processing_functions import get_reduced_tensors_from_level, get_full_tensors_from_level
import Constants
from stateSpaceFunctions import get_block_sequence_with_deltas

class StageTwoDataset(BaseDataset):

    def __init__(self, opt,receptive_field=None):
        super().__init__()
        self.opt = opt
        self.receptive_field = receptive_field
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError('Invalid directory:'+opt.data_dir)
        # self.audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        candidate_audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        self.level_jsons = []
        self.audio_files = []
        self.feature_files = {}

        for i, path in enumerate(candidate_audio_files):
            #print(path)
            features_file = path.__str__()+"_"+self.opt.feature_name+"_"+str(self.opt.feature_size)+".npy"
            level_file_found = False
            for diff in self.opt.level_diff.split(","):
                if Path(path.parent.__str__()+"/"+diff+".json").is_file():
                    level_file_found = True
            if not level_file_found:
                continue

            # we need to find out what the input length of the model is, to remove songs which are too short to get input windows from them for this model
            receptive_field = self.receptive_field
            output_length = self.opt.output_length
            input_length = receptive_field + output_length -1
            try:
                features = np.load(features_file)

                if (features.shape[1]-(input_length+self.opt.time_shifts-1)) < 1:
                    print("Smol song; ignoring..")
                    continue

                self.feature_files[path.__str__()] = features_file
            except FileNotFoundError:
                raise Exception("An unprocessed song found; need to run preprocessing script process_songs.py before starting to train with them")

            for diff in self.opt.level_diff.split(","):
                try:
                    level = list(path.parent.glob('./'+diff+'.json'))[0]
                    self.level_jsons.append(level)
                    self.audio_files.append(path)
                except:
                    continue


        assert self.audio_files, "List of audio files cannot be empty"
        assert self.level_jsons, "List of level files cannot be empty"
        assert len(self.audio_files) == len(self.level_jsons)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=16000, type=float)
        parser.add_argument('--level_diff', default='Expert', help='Difficulty level for beatsaber level')
        parser.add_argument('--hop_length', default=256, type=int)  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        parser.add_argument('--compute_feats', action='store_true', help="Whether to extract musical features from the song")
        parser.add_argument('--padded_length', type=int, default=3000000)
        parser.add_argument('--chunk_length', type=int, default=9000)
        parser.add_argument('--feature_name', default='chroma')
        parser.add_argument('--feature_size', type=int, default=24)
        parser.add_argument('--time_shifts', type=int, default=1, help='number of shifted sequences to include as input')
        parser.add_argument('--reduced_state', action='store_true', help='if true, use reduced state representation')
        parser.add_argument('--concat_outputs', action='store_true', help='if true, concatenate the outputs to the input sequence')
        parser.add_argument('--extra_output', action='store_true', help='set true for wavenet, as it needs extra output to predict, other than the outputs fed as input :P')
        parser.add_argument('--binarized', action='store_true', help='set true to predict only wheter there is a state or not')
        parser.add_argument('--max_token_seq_len', type=int, default=1000)
        parser.set_defaults(output_length=1)

        return parser

    def name(self):
        return "stage_two_dataset"

    def __getitem__(self, item):
        #NOTE: there is a lot of code repeat between this and the non-reduced version, perhaps we could fix that
        song_file_path = self.audio_files[item].__str__()
        # print(song_file_path)

        level = json.load(open(self.level_jsons[item].__str__(), 'r'))

        bpm = level['_beatsPerMinute']
        features_rate = bpm*self.opt.beat_subdivision
        notes = level['_notes']

        #useful quantities, to sync notes to song features
        sr = self.opt.sampling_rate
        if self.opt.using_bpm_time_division:
            beat_subdivision = self.opt.beat_subdivision
            beat_duration = 60/bpm #beat duration in seconds
            sample_duration = step_size = beat_duration/beat_subdivision #in seconds
        else:
            sample_duration = step_size = self.opt.step_size # in seconds
        features = np.load(self.feature_files[song_file_path])


        # for short
        y = features

        receptive_field = self.receptive_field
        # we pad the song features with zeros to imitate during training what happens during generation
        # this is helpful for models that have a big receptive field like wavent, but we also use it with a receptive_field=1 for LSTM and Transformer
        y = np.concatenate((np.zeros((y.shape[0],receptive_field)),y),1)
        # we also pad one more state at the end, to accommodate an "end" symbol for the blocks
        y = np.concatenate((y,np.zeros((y.shape[0],1))),1)

        sequence_length = y.shape[1]*sample_duration

        ## BLOCKS TENSORS ##
        one_hot_states, states, delta_forward, delta_backward, indices = get_block_sequence_with_deltas(self.level_jsons[item].__str__(),sequence_length,bpm,step_size, top_k=2000,states=unique_states,one_hot=True)
        # print(indices.shape,states.shape,one_hot_states.shape,delta_forward.shape,delta_backward.shape)
        truncated_sequence_length = min(len(states),self.opt.max_token_seq_len)
        states = states[:truncated_sequence_length]
        indices = indices[:truncated_sequence_length]
        one_hot_states = one_hot_states[:,:truncated_sequence_length]
        delta_forward = delta_forward[:,:truncated_sequence_length]
        delta_backward = delta_backward[:,:truncated_sequence_length]

        target_block_sequence = torch.tensor(states).unsqueeze(0).unsqueeze(1).long()
        input_forward_deltas = torch.tensor(delta_forward).unsqueeze(0).long()
        input_backward_deltas = torch.tensor(delta_backward).unsqueeze(0).long()

        # get features at the places where a note appears, to construct feature sequence to help transformer
        y = y[:,indices]
        print(y.shape)
        input_windows = [y]

        song_sequence = torch.tensor(input_windows)
        song_sequence = (song_sequence - song_sequence.mean())/torch.abs(song_sequence).max().float()

        ## vv if we fed deltas as decoder transformer input :P
        if self.opt.tgt_vector_input:
            input_block_sequence = torch.tensor(one_hot_states).unsqueeze(0).long()
            input_block_deltas = torch.cat([input_block_sequence,input_forward_deltas,input_backward_deltas],1)
            return {'input': song_sequence, 'target': torch.cat([target_block_sequence,input_block_deltas],1)}
        else:
            song_sequence = torch.cat([song_sequence,input_forward_deltas.double(),input_backward_deltas.double()],1)
            return {'input': song_sequence, 'target': target_block_sequence}

    def __len__(self):
        return len(self.audio_files)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
