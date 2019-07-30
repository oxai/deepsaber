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
from featureExtration import extract_features_hybrid, extract_features_mel,extract_features_hybrid_beat_synced
import Constants

class GeneralBeatSaberDataset(BaseDataset):

    def __init__(self, opt,receptive_field=None):
        super().__init__()
        self.opt = opt
        self.receptive_field = receptive_field
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError('Invalid directory:'+opt.data_dir)
        candidate_audio_files = sorted(data_path.glob('**/*.egg'), key=lambda path: path.parent.__str__())
        self.level_jsons = []
        self.info_jsons = []
        self.audio_files = []
        self.feature_files = {}

        # cleaning up data, from songs which are too short
        for i, path in enumerate(candidate_audio_files):
            #print(path)
            features_file = path.__str__()+"_"+self.opt.feature_name+"_"+str(self.opt.feature_size)+".npy"
            level_file_found = False
            for diff in self.opt.level_diff.split(","):
                if Path(path.parent.__str__()+"/"+diff+".dat").is_file():
                    level_file_found = True
            if not level_file_found:
                continue

            # we need to find out what the input length of the model is, to remove songs which are too short to get input windows from them for this model
            receptive_field = self.receptive_field
            output_length = self.opt.output_length
            input_length = receptive_field + output_length -1
            try:
                features = np.load(features_file)
                if (features.shape[-1]-(input_length+self.opt.time_shifts-1)) < 1:
                    print("Smol song; ignoring..")
                    continue
                self.feature_files[path.__str__()] = features_file
            except FileNotFoundError:
                raise Exception("An unprocessed song found; need to run preprocessing script process_songs.py before starting to train with them")

            for diff in self.opt.level_diff.split(","):
                try:
                    level = list(path.parent.glob('./'+diff+'.dat'))[0]
                    info_file = list(path.parent.glob('./info.dat'))[0]
                    self.level_jsons.append(level)
                    self.info_jsons.append(info_file)
                    self.audio_files.append(path)
                except:
                    continue


        assert self.audio_files, "List of audio files cannot be empty"
        assert self.level_jsons, "List of level files cannot be empty"
        assert self.info_jsons, "List of info files cannot be empty"
        assert len(self.audio_files) == len(self.level_jsons)
        assert len(self.audio_files) == len(self.info_jsons)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=44100, type=float)
        parser.add_argument('--level_diff', default='Expert', help='Difficulty level for beatsaber level')
        parser.add_argument('--feature_name', default='chroma')
        parser.add_argument('--feature_size', type=int, default=24)
        parser.add_argument('--hop_length', default=256, type=int)  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        parser.add_argument('--compute_feats', action='store_true', help="Whether to extract musical features from the song")
        parser.add_argument('--padded_length', type=int, default=3000000)
        parser.add_argument('--chunk_length', type=int, default=9000)
        # the input features at each time step consiste of the features at the time steps from now to time_shifts in the future
        parser.add_argument('--time_shifts', type=int, default=1, help='number of shifted sequences to include as input')
        parser.add_argument('--output_length', type=int, default=1)
        parser.add_argument('--time_offset', type=int, default=1, help='time shift between the last read input, and the output predicted. The default value of 1 corresponds to predicting the next output')
        parser.add_argument('--reduced_state', action='store_true', help='if true, use reduced state representation')
        parser.add_argument('--concat_outputs', action='store_true', help='if true, concatenate the outputs to the input sequence')
        parser.add_argument('--extra_output', action='store_true', help='set true for wavenet, as it needs extra output to predict, other than the outputs fed as input :P')
        parser.add_argument('--binarized', action='store_true', help='set true to predict only wheter there is a state or not')
        parser.add_argument('--flatten_context', action='store_true', help='whether to flatten the temporal context added for each time point into the feature dimension, or not')
        parser.add_argument('--max_token_seq_len', type=int, default=1024)
        parser.set_defaults(output_length=1)
        parser.set_defaults(output_channels=1)

        return parser

    def name(self):
        return "GeneralBeatSaberDataset"

    def __getitem__(self, item):
        song_file_path = self.audio_files[item].__str__()

        level = json.load(open(self.level_jsons[item].__str__(), 'r'))
        info = json.load(open(self.info_jsons[item].__str__(), 'r'))

        bpm = info['_beatsPerMinute']
        features_rate = bpm*self.opt.beat_subdivision
        notes = level['_notes']

        #useful quantities, to sync notes to song features
        sr = self.opt.sampling_rate
        if self.opt.using_bpm_time_division:
            beat_duration_samples = int(60*sr/bpm) #beat duration in samples
            beat_subdivision = self.opt.beat_subdivision
            hop = int(beat_duration_samples * 1/beat_subdivision)
            beat_duration = 60/bpm #beat duration in seconds
            step_size = beat_duration/beat_subdivision #in seconds
        else:
            step_size = self.opt.step_size # in seconds
            hop = int(step_size*sr)
            beat_subdivision = 1/(step_size*bpm/60)
        # duration of one time step in samples:
        num_samples_per_feature = hop
        features = np.load(self.feature_files[song_file_path])

        # for short
        y = features #dimensions of y are: features x time OR features x window_sizes x time
        print(y.shape)

        receptive_field = self.receptive_field
        # we pad the song features with zeros to imitate during training what happens during generation
        # this is helpful for models that have a big receptive field like wavent, but we also use it with a receptive_field=1 for LSTM and Transformer
        if len(y.shape) == 2: # one feature dimension in y
            y = np.concatenate((np.zeros((y.shape[0],receptive_field+self.opt.time_shifts//2)),y),1)
            # we also pad at the end to allow generation to be of the same length of song, by padding an amount corresponding to time_shifts
            y = np.concatenate((y,np.zeros((y.shape[0],self.opt.time_shifts//2))),1)
        elif len(y.shape) == 3: #two feature dimensions in y
            y = np.concatenate((np.zeros((y.shape[0],y.shape[1],receptive_field+self.opt.time_shifts//2)),y),2)
            # we also pad at the end to allow generation to be of the same length of song, by padding an amount corresponding to time_shifts
            y = np.concatenate((y,np.zeros((y.shape[0],y.shape[1],self.opt.time_shifts//2))),2)

        ## WINDOWS ##
        # sample indices at which we will get opt.num_windows windows of the song to feed as inputs
            # TODO: make this deterministic, and determined by `item`, so that one epoch really corresponds to going through all the data..
        sequence_length = min(y.shape[-1],self.opt.max_token_seq_len)
        output_length = self.opt.output_length
        time_offset = self.opt.time_offset
        if self.opt.num_windows >= 1:
            input_length = receptive_field + self.opt.output_length - 1
            if self.opt.model == "wavenet": #because blocks have a causal auto-regressive receptive field
                time_offset = -receptive_field//2
                blocks_receptive_field = receptive_field//2
                blocks_input_length = blocks_receptive_field + self.opt.output_length - 1
                blocks_time_offset = 0
            else:
                blocks_receptive_field = receptive_field
                blocks_input_length = input_length
                blocks_time_offset = time_offset
            # input_length = self.opt.input_length
            indices = np.random.choice(range(self.opt.time_shifts//2,sequence_length-(input_length+self.opt.time_shifts//2+self.opt.time_offset)),size=self.opt.num_windows,replace=True)
        elif self.opt.time_shifts == 1:
            indices=np.array([0])
            input_length = sequence_length
        else:
            raise Exception("Can't have time_shifts with setting num_windows=0 (which means take the whole sequence)")

        ## CONSTRUCT TENSOR OF INPUT SOUND FEATURES ##
        if self.opt.time_shifts >= 1: #if we are including context
            input_windowss = []
            time_shifts = self.opt.time_shifts - 1
            # loop that gets the input features for each of the windows, shifted by `ii`, and saves them in `input_windowss`
            for ii in range(-time_shifts//2, time_shifts//2+1):
                if len(y.shape) == 2:
                    input_windows = [y[:,i+ii:i+ii+input_length] for i in indices]
                elif len(y.shape) == 3:
                    input_windows = [y[:,:,i+ii:i+ii+input_length] for i in indices]
                input_windows = torch.tensor(input_windows)
                input_windows = (input_windows - input_windows.mean())/torch.abs(input_windows).max()
                input_windowss.append(input_windows.float())
        else: #if we are not including context
            if len(y.shape) == 2:
                input_windows = [y[:,i:i+input_length] for i in indices]
            elif len(y.shape) == 3:
                input_windows = [y[:,:,i:i+input_length] for i in indices]
            input_windows = torch.tensor(input_windows)
            input_windows = (input_windows - input_windows.mean())/torch.abs(input_windows).max()
            input_windowss = [input_windows]
        # print(len(y.shape))

        ## BLOCKS TENSORS ##
        print(input_length, output_length, time_offset)
        if self.opt.reduced_state:
            blocks_windows, blocks_targets = get_reduced_tensors_from_level(notes,indices,sequence_length,self.opt.num_classes,bpm,sr,num_samples_per_feature,blocks_receptive_field,blocks_input_length, output_length,blocks_time_offset)
            print("blocks_targets",blocks_targets.shape)
            print("blocks_windows",blocks_windows.shape)
            print("input_windowss",input_windowss[0].shape)
        elif self.opt.binarized:
            blocks_windows, blocks_targets = get_reduced_tensors_from_level(notes,indices,sequence_length,1+Constants.NUM_SPECIAL_STATES,bpm,sr,num_samples_per_feature,blocks_receptive_field,blocks_input_length, output_length,blocks_time_offset)
        else: #not tested
            blocks_windows, blocks_targets = get_full_tensors_from_level(notes,indices,sequence_length,self.opt.num_classes,self.opt.output_channels,bpm,sr,num_samples_per_feature,receptive_field,input_length)

        if self.opt.concat_outputs: #for autoregressive models. If we want to concatenate the outputs to the inputs too.
            if self.opt.flatten_context: #if we want to flatten the feature vectors given as context via time-shifts
                return {'input': torch.cat(input_windowss + [blocks_windows.float()],1).float(), 'target': blocks_targets}
            else:
                raise NotImplementedError("Haven't implemented non-flattened context, when concat_outputs")
                # return {'input': torch.cat(np.stack(input_windowss,axis=1) + np.stack(blocks_windowss,axis=1),2).float(), 'target': blocks_targets}
        else:
            if self.opt.flatten_context: #if we want to flatten the feature vectors given as context via time-shifts
                return {'input': torch.cat(input_windowss,1), 'target': blocks_targets}
            else:
                return {'input': torch.stack(input_windowss,dim=1).float(), 'target': blocks_targets}

    def __len__(self):
        return len(self.audio_files)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
