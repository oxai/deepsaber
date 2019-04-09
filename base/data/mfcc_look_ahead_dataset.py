from pathlib import Path
from itertools import tee
import numpy as np
import torch
import librosa
from base.data.base_dataset import BaseDataset
import json
from math import floor, ceil
import pickle

class MfccLookAheadDataset(BaseDataset):

    def __init__(self, opt,receptive_field=None):
        super().__init__()
        self.opt = opt
        self.receptive_field = receptive_field
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError(f'Invalid directory: {opt.data_dir}')
        # self.audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        candidate_audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        self.level_jsons = []
        self.audio_files = []
        self.mfcc_features = {}
        for i, path in enumerate(candidate_audio_files):
            #print(path)
            try:
                level = list(path.parent.glob('./'+self.opt.level_diff+'.json'))[0]
                self.level_jsons.append(level)
                self.audio_files.append(path)
            except IndexError:
                continue
            
            n_mfcc=(opt.input_channels - opt.output_channels*opt.num_classes)//opt.time_shifts             
            mfcc_file = path.__str__()+"_"+str(n_mfcc)+"_"+str(self.opt.beat_subdivision)+"_mfcc.npy"
            try:
                # mfcc = pickle.load(open(mfcc_file,"rb"))
                mfcc = np.load(mfcc_file)
                #print("reading mfcc file")
                
                # we need to find out what the input length of the model is, to remove songs which are too short to get input windows from them for this model
                receptive_field = self.receptive_field
                output_length = self.opt.output_length
                input_length = receptive_field + output_length -1

                if mfcc.shape[1]-(input_length+self.opt.time_shifts-1) < 1:
                    print("Smol song; ignoring..")
                    self.level_jsons.pop()
                    self.audio_files.pop()
                    continue

                self.mfcc_features[path.__str__()] = mfcc
            except FileNotFoundError:
                raise Exception("An unprocessed song found; need to run preprocessing script process_songs.py before starting to train with them")

        assert self.audio_files, "List of audio files cannot be empty"
        assert self.level_jsons, "List of level files cannot be empty"
        assert len(self.audio_files) == len(self.level_jsons)
        self.eps = 0.1

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=16000, type=float)
        parser.add_argument('--level_diff', default='Expert', help='Difficulty level for beatsaber level')
        parser.add_argument('--hop_length', default=256, type=int)  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        parser.add_argument('--compute_feats', action='store_true', help="Whether to extract musical features from the song")
        parser.add_argument('--padded_length', type=int, default=3000000)
        parser.add_argument('--chunk_length', type=int, default=9000)
        # the input features at each time step consiste of the mfcc features at the time steps from now to time_shifts in the future 
        parser.add_argument('--time_shifts', type=int, default=16)
        # the total number of input_channels is constructed by the the nfcc features (20 of them), 16 times one for each time_shift as explained above
        # plus 12*20 
        # corresponding to 12 points in the grid, and one of 20 classes at each point
        # for each point in the grid we have a one-hot vector of size 20
        # and we just concatenate these 12 one-hot vectors
        # to get a size 12*20 "many-hot" vector
        parser.set_defaults(input_channels=(20*16+12*20))
        # there are 12 outputs, one per grid point, with 20 possible classes each.
        parser.set_defaults(output_channels=12)
        parser.set_defaults(num_classes=20)
        return parser

    def name(self):
        return "SongDataset"

    def __getitem__(self, item):
        song_file_path = self.audio_files[item].__str__()
        mfcc_file = song_file_path+"_"+str(self.opt.beat_subdivision)+"_mfcc.npy"

        # get mfcc features
        try:
            mfcc = self.mfcc_features[song_file_path]
        except:
            raise Exception("mfcc features not found for song "+song_file_path)

        level = json.load(open(self.level_jsons[item].__str__(), 'r'))

        bpm = level['_beatsPerMinute']
        features_rate = bpm*self.opt.beat_subdivision
        notes = level['_notes']

        sr = self.opt.sampling_rate
        beat_duration = int(60*sr/bpm) #beat duration in samples
        # duration of one time step in samples:
        mel_hop = beat_duration//self.opt.beat_subdivision #this is the number of samples between successive mfcc frames (as used in the data processing file), so I think that means each frame occurs every mel_hop + 1. I think being off by one sound sample isn't a big worry though.
        num_samples_per_feature = mel_hop + 1 

        # for short
        y = mfcc

        ## WINDOWS ##
        # sample indices at which we will get opt.num_windows windows of the song to feed as inputs
        # TODO: make this deterministic, and determined by `item`, so that one epoch really corresponds to going through all the data..
        receptive_field = self.receptive_field
        output_length = self.opt.output_length
        input_length = receptive_field + output_length -1
        indices = np.random.choice(range(y.shape[1]-(input_length+self.opt.time_shifts-1)),size=self.opt.num_windows,replace=True)

        ## CONSTRUCT TENSOR OF INPUT SOUND FEATURES (MFCC) ##
        # loop that gets the input features for each of the windows, shifted by `ii`, and saves them in `input_windowss`
        input_windowss = []
        for ii in range(self.opt.time_shifts):
            input_windows = [y[:,i+ii:i+ii+input_length] for i in indices]
            input_windows = torch.tensor(input_windows)
            input_windows = (input_windows - input_windows.mean())/torch.abs(input_windows).max()
            input_windowss.append(input_windows.float())

        ## BLOCKS TENSORS ##
        # variable `blocks` of shape (time steps, number of locations in the block grid), storing the class of block (as a number from 0 to 19) at each point in the grid, at each point in time
        blocks = np.zeros((y.shape[1],self.opt.output_channels)) 
        #many-hot vector
        # for each point in the grid we will have a one-hot vector of size 20 (num_classes)
        # and we will just stack these 12 (output_channels) one-hot vectors
        # to get a "many-hot" tensor of shape (time_steps,output_channels,num_classes)
        blocks_manyhot = np.zeros((y.shape[1],self.opt.output_channels,self.opt.num_classes)) 
        #we initialize the one-hot vectors in the tensor
        blocks_manyhot[:,:,0] = 1.0 #default is the "nothing" class

        ## CONSTRUCT BLOCKS TENSOR ##
        for note in notes:
            #sample_index = floor((time of note in seconds)*sampling_rate/(num_samples_per_feature))
            sample_index = floor((note['_time']*60/bpm)*sr/num_samples_per_feature)
            # check if note falls within the length of the song (why are there so many that don't??) #TODO: research why this happens
            if sample_index >= y.shape[1]:
                print("note beyond the end of time")
                continue

            #constructing the representation of the block (as a number from 0 to 19)
            if note["_type"] == 3:
                note_representation = 19
            elif note["_type"] == 0 or note["_type"] == 1:
                note_representation = 1 + note["_type"]*9+note["_cutDirection"]
            else:
                raise ValueError("I thought there was no notes with _type different from 0,1,3. Ahem, what are those??")
            blocks[sample_index,note["_lineLayer"]*4+note["_lineIndex"]] = note_representation
            blocks_manyhot[sample_index,note["_lineLayer"]*4+note["_lineIndex"], 0] = 0.0 #remove the one hot at the zero class
            blocks_manyhot[sample_index,note["_lineLayer"]*4+note["_lineIndex"], note_representation] = 1.0


        # get the block features corresponding to the windows
        block_windows = [blocks[i+receptive_field:i+input_length+1,:] for i in indices]
        block_windows = torch.tensor(block_windows,dtype=torch.long)

        blocks_manyhot_windows = [blocks_manyhot[i:i+input_length,:,:] for i in indices]
        blocks_manyhot_windows = torch.tensor(blocks_manyhot_windows)
        # this is because the input features have dimensions (num_windows,time_steps,num_features)
        blocks_manyhot_windows = blocks_manyhot_windows.permute(0,2,3,1)
        shape = blocks_manyhot_windows.shape
        # now we reshape so that the stack of one-hot vectors becomes a single "many-hot" vector
        # formed by concatenating the one hot vectors
        blocks_manyhot_windows = blocks_manyhot_windows.view(shape[0],shape[1]*shape[2],shape[3]).float()

        # concatenate the song and block input features before returning
        return {'input': torch.cat(input_windowss + [blocks_manyhot_windows],1), 'target': block_windows}

    def __len__(self):
        return len(self.audio_files)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
