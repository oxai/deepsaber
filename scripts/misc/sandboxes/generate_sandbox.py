import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)
import argparse
import time
from models import create_model
import json, pickle
import torch
import numpy as np
import models.constants as constants
from math import ceil
from scipy import signal

from scripts.generation.level_generation_utils import extract_features, make_level_from_notes, get_notes_from_stepmania_file

parser = argparse.ArgumentParser(description='Generate Beat Saber level from song')
parser.add_argument('--song_path', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--experiment_name2', type=str, default=None)
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--checkpoint2', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)
parser.add_argument('--generate_full_song', action="store_true")
parser.add_argument('--use_beam_search', action="store_true")
parser.add_argument('--two_stage', action="store_true")
parser.add_argument('--use_ddc', action="store_true")
parser.add_argument('--ddc_file', type=str, default='test_ddc.sm')
parser.add_argument('--ddc_diff', type=int, default=1)
parser.add_argument('--open_in_browser', action="store_true")

args = parser.parse_args()

# debugging helpers

# checkpoint = "64000"
# checkpoint = "330000"
# checkpoint2 = "68000"
# temperature = 1.00
# experiment_name = "block_placement/"
# experiment_name2 = "block_selection/"
# two_stage = True
# args={}
# args["checkpoint"] = "975000"
# # args["checkpoint"] = "125000"
# args["checkpoint2"] = "875000"
# args["experiment_name"] = "block_placement_dropout/"
# args["experiment_name"] = "block_placement_new_nohumreg/"
# # args["experiment_name"] = "block_placement_new_nohumreg_small/"
# # args["experiment_name"] = "block_placement_new_nohumreg_large/"
# args["experiment_name2"] = "block_selection_new/"
# args["temperature"] = 1.00
# args["peak_threshold"] = 0.0148
# args["two_stage"] = True
# args["bpm"] = None
# args["use_ddc"] = False
# args["ddc_file"] = "test_ddc.sm"
# args["generate_full_song"] = False
# args["use_beam_search"] = True
# args["open_in_browser"] = True

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
song_path=args.song_path
ddc_file=args.ddc_file
if args.two_stage:
    assert args.experiment_name2 is not None
    assert args.checkpoint2 is not None

from pathlib import Path
song_name = Path(song_path).stem

''' LOAD MODEL, OPTS, AND WEIGHTS (for stage1 if two_stage) '''
#%%

##loading opt object from experiment
opt = json.loads(open("../training/"+experiment_name+"opt.json","r").read())
# we assume we have 1 GPU in generating machine :P
opt["gpu_ids"] = [0]
opt["load_iter"] = int(checkpoint)
opt["cuda"] = True
opt["experiment_name"] = args.experiment_name.split("/")[0]
if "dropout" not in opt: #for older experiments
    opt["dropout"] = 0.0
# construct opt Struct object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

if args.two_stage:
    assert opt.binarized

if not args.use_ddc:
    model = create_model(opt)
    model.setup()
    if opt.model=='wavenet' or opt.model=='adv_wavenet':
        if not opt.gpu_ids:
            receptive_field = model.net.receptive_field
        else:
            receptive_field = model.net.module.receptive_field
    else:
        receptive_field = 1

    checkpoint = "iter_"+checkpoint
    model.load_networks(checkpoint)

''' GET SONG FEATURES '''
#%%

hop, features = extract_features(song_path, args, opt)


''' GENERATE LEVEL '''
#%%
if not args.use_ddc: #if using DDC first stage
    song = torch.tensor(features).unsqueeze(0)

    #generate level
    # first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
    first_samples = torch.full((1,opt.output_channels,receptive_field//2),constants.START_STATE)
    # stuff for older models (will remove at some point:)
    ## first_samples = torch.full((1,opt.output_channels,receptive_field),constants.EMPTY_STATE)
    ## first_samples[0,0,0] = constants.START_STATE
    print("Generating level timings... (sorry I'm a bit slow)")
    if opt.concat_outputs: #whether to concatenate the generated outputs as new inputs (AUTOREGRESSIVE)
        output,peak_probs = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
        peak_probs = np.array(peak_probs)

        window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
        smoothed_peaks = np.convolve(peak_probs,window,mode='same')
        index = np.random.randint(len(smoothed_peaks))
        # for debugg help, but maybe useful for future work too
        # import matplotlib.pyplot as plt
        # plt.plot(smoothed_peaks[index:index+1000])

        # plt.savefig("smoothed_peaks.png")
        # pickle.dump(smoothed_peaks, open("smoothed_peaks.p","wb"))
        thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
        peaks = signal.find_peaks(thresholded_peaks)[0]
        print("number of peaks", len(peaks))
    else: # NOT-AUTOREGRESSIVE (we keep it separate like this, because some models have both)
        output = model.net.module.generate_no_autoregressive(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
    if not opt.binarized:
        states_list = output[0,:,:].permute(1,0)
#%%
else: # if not using DDC first stage
    diff = args.ddc_diff
    print("Reading ddc file ", ddc_file)
    notes = get_notes_from_stepmania_file(ddc_file, diff)
    times_real = [(4*(60/125)/192)*note for note in notes]
    # notes
#%%

#convert from states to beatsaber notes
print("Processing notes...")
if opt.binarized: # for experiments where the output is state/no state
    if not args.use_ddc:
        times_real = [float(i*hop/opt.sampling_rate) for i in peaks]
    notes = [{"_time":float(t*args.bpm/60), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0} for t in times_real]
    print("Number of generated notes: ", len(notes))
    notes = np.array(notes)[np.where(np.diff([-1]+times_real) > constants.HUMAN_DELTA)[0]].tolist()
else: # this is where the notes are generated for end-to-end models that actually output states
    unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
    states_list = [(unique_states[i[0].int().item()-4] if i[0].int().item() not in [0,1,2,3] else tuple(12*[0])) for i in states_list ]
    notes = [[{"_time":float((i+0.0)*args.bpm*hop/(opt.sampling_rate*60)), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
    notes += [[{"_time":float((i+0.0)*args.bpm*hop/(opt.sampling_rate*60)), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
    notes = sum(notes,[])

print("Number of generated notes (after pruning): ", len(notes))

json_file = make_level_from_notes(notes, args.bpm, song_name, opt, args)

#%%

''' STAGE TWO! '''

if args.two_stage:
    print("STAGE TWO!")
    #%%
    ''' LOAD MODEL, OPTS, AND WEIGHTS (for stage2 if two_stage) '''
    experiment_name = args.experiment_name2+"/"
    checkpoint = args.checkpoint2

    #loading opt object from experiment, and constructing Struct object after adding some things
    opt = json.loads(open("../training/"+experiment_name+"opt.json","r").read())
    # extra things Beam search wants
    opt["gpu_ids"] = [0]
    opt["load_iter"] = int(checkpoint)
    opt["cuda"] = True
    opt["batch_size"] = 1
    opt["beam_size"] = 20
    opt["n_best"] = 1
    # opt["using_bpm_time_division"] = True
    opt["continue_train"] = False
    # opt["max_token_seq_len"] = len(notes)
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    opt = Struct(**opt)

    model = create_model(opt)
    model.setup()
    checkpoint = "iter_"+checkpoint
    model.load_networks(checkpoint)
    unique_states = pickle.load(open("../../data/statespace/sorted_states.pkl","rb"))

    ''' GET SONG FEATURES for stage two '''
    #%%

    hop, features = extract_features(song_path, args, opt)

    #%%
    print("Generating state sequence...")
    state_times, generated_sequence = model.generate(features, json_file, args.bpm, unique_states, temperature=temperature, use_beam_search=args.use_beam_search, generate_full_song=args.generate_full_song)
    # state_times is the times of the nonemtpy states, in bpm units

    #%%
    from scripts.data_processing.state_space_functions import stage_two_states_to_json_notes
    times_real = [t*60/args.bpm for t in state_times]
    notes2 = stage_two_states_to_json_notes(generated_sequence, state_times, args.bpm, hop, opt.sampling_rate, state_rank=unique_states)
    # print("Bad notes:", np.unique(np.diff(times_real)[np.diff(times_real)<=constants.HUMAN_DELTA], return_counts=True))

    make_level_from_notes(notes2, args.bpm, song_name, opt, args, upload_to_dropbox=True, open_in_browser=args.open_in_browser)
