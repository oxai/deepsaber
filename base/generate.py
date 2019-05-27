import argparse
import sys, os, time
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber/base/models")
sys.path.append("/home/guillefix/code/beatsaber")
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
import json, pickle
import librosa
import torch
import numpy as np
import Constants
from level_generation_utils import make_level_from_notes

from stateSpaceFunctions import feature_extraction_hybrid_raw,feature_extraction_mel,feature_extraction_hybrid

# parser = argparse.ArgumentParser(description='Generate Beat Saber level from song')
# parser.add_argument('--experiment_name', type=str)
# parser.add_argument('--experiment_name2', type=str, default=None)
# parser.add_argument('--checkpoint', type=str, default="latest")
# parser.add_argument('--checkpoint2', type=str, default="latest")
# parser.add_argument('--temperature', type=float, default=1.00)
# parser.add_argument('--bpm', type=float, default=None)
# parser.add_argument('--two_stage', action="store_true")
#
# args = parser.parse_args()


# debugging helpers

# checkpoint = "64000"
# checkpoint = "330000"
# checkpoint2 = "68000"
# temperature = 1.00
# experiment_name = "block_placement/"
# experiment_name2 = "block_selection/"
# two_stage = True
args={}
args["checkpoint"] = "60000"
args["checkpoint2"] = "15000"
args["experiment_name"] = "block_placement_new/"
args["experiment_name2"] = "block_selection_new/"
args["temperature"] = 1.00
args["two_stage"] = True
args["bpm"] = None
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = Struct(**args)

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
if args.two_stage:
    assert args.experiment_name2 is not None
    assert args.checkpoint2 is not None

song_name = "35_fixed"
song_name = "test_song"+song_name+".wav"
song_path = "../../"+song_name
# print(experiment_name)

''' LOAD MODEL, OPTS, AND WEIGHTS (for stage1 if two_stage) '''
#%%

#loading opt object from experiment
opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
opt["cuda"] = True
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

if args.two_stage:
    assert opt.binarized

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

y_wav, sr = librosa.load(song_path, sr=16000)

from test_song_bpms import bpms

# useful quantities
if args.bpm is not None:
    bpm = args.bpm
else:
    bpm = bpms[song_name]
feature_name = opt.feature_name
feature_size = opt.feature_size
sampling_rate = opt.sampling_rate
beat_subdivision = opt.beat_subdivision
try:
    step_size = opt.step_size
    using_bpm_time_division = opt.using_bpm_time_division
except: # older model
    using_bpm_time_division = True

sr = sampling_rate
beat_duration = 60/bpm #beat duration in seconds
beat_duration_samples = int(60*sr/bpm) #beat duration in samples
if using_bpm_time_division:
    # duration of one time step in samples:
    hop = int(beat_duration_samples * 1/beat_subdivision)
    # num_samples_per_feature = hop
    step_size = beat_duration/beat_subdivision # in seconds
else:
    beat_subdivision = 1/(step_size*bpm/60)
    hop = step_size*sr

# get feature
sample_times = np.arange(0,y_wav.shape[0]/sr,step=step_size)
if opt.feature_name == "chroma":
    features = feature_extraction_hybrid(y_wav,sr,sample_times,bpm,beat_discretization=1/beat_subdivision,mel_dim=12)
elif opt.feature_name == "mel":
    features = feature_extraction_mel(y_wav,sr,sample_times,bpm,mel_dim=feature_size,beat_discretization=1/beat_subdivision)
    features = librosa.power_to_db(features, ref=np.max)


''' GENERATE LEVEL '''
#%%
song = torch.tensor(features).unsqueeze(0)
# temperature = 1.00

#generate level
first_samples = torch.full((1,opt.output_channels,receptive_field//2),Constants.START_STATE)
# first_samples = torch.full((1,receptive_field//2),Constants.START_STATE)
# stuff for older models (will remove at some point:)
# first_samples = torch.full((1,opt.output_channels,receptive_field),Constants.EMPTY_STATE)
# first_samples[0,0,0] = Constants.START_STATE
if opt.concat_outputs:
    output = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
else:
    output = model.net.module.generate_no_autoregressive(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
states_list = output[0,:,:].permute(1,0)
states_list.tolist()
np.unique(states_list)

#if using reduced_state representation convert from reduced_state_index to state tuple
unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
#old (before transformer)
# states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]

#convert from states to beatsaber notes
if opt.binarized: # for experiments where the output is state/no state
    notes = [{"_time":float((i-1)*bpm*hop/(sr*60)), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0} for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
    # times_beat = [float((i+0.0)*bpm*hop/(sr*60)) for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
    times_real = [float((i-1)*hop/sr) for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
    notes = np.array(notes)[np.where(np.diff([-1]+times_real) > 0.1)[0]].tolist()
else: # this is where the notes are generated for end-to-end models that actually output states
    states_list = [(unique_states[i[0].int().item()-4] if i[0].int().item() not in [0,1,2,3] else tuple(12*[0])) for i in states_list ]
    notes = [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
    notes += [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
    notes = sum(notes,[])

print("Number of generated notes: ", len(notes))

json_file = make_level_from_notes(notes, bpm, song_name, opt, args)
# notes
# list(map(lambda x: ))
# times = [note["_time"] for note in notes]
# np.unique(np.diff(times_real), return_counts=True)
# np.diff(times_real) <= 0.125
# np.diff(times) <= 0.125
# len(times) = len()
json_file = make_level_from_notes(notes, bpm, song_name, opt, args, open_in_browser=True)


#%%

''' STAGE TWO! '''

if args.two_stage:
    #%%
    ''' LOAD MODEL, OPTS, AND WEIGHTS (for stage2 if two_stage) '''
    experiment_name = args.experiment_name2+"/"
    checkpoint = args.checkpoint2

    #loading opt object from experiment
    opt = json.loads(open(experiment_name+"opt.json","r").read())
    # extra things Beam search wants
    opt["gpu_ids"] = [0]
    opt["cuda"] = True
    opt["batch_size"] = 1
    opt["beam_size"] = 5
    opt["n_best"] = 5
    opt["using_bpm_time_division"] = True
    opt["continue_train"] = False
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    opt = Struct(**opt)

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

    # generated_folder = "generated/"
    # signature_string = song_name+"_"+opt.model+"_"+opt.dataset_name+"_"+opt.experiment_name+"_"+str(temperature)+"_"+args.checkpoint
    # signature_string = song_name+"_"+"wavenet"+"_"+"general_beat_saber"+"_"+"block_placement"+"_"+str(temperature)+"_"+args.checkpoint
    # json_file = generated_folder+"test_song"+signature_string+".json"
    # json_file = "/home/guillefix/code/beatsaber/DataE/156)Rap God (Explicit) - /Rap God/ExpertPlus.json"
    # sequence_length = 366
    # from stateSpaceFunctions import get_block_sequence_with_deltas
    # one_hot_states, states, state_times, delta_forward, delta_backward, indices = get_block_sequence_with_deltas(json_file,sequence_length,bpm,top_k=2000,beat_discretization=1/opt.beat_subdivision,states=unique_states,one_hot=True,return_state_times=True)

    unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))

    #%%
    # import imp; import stateSpaceFunctions; imp.reload(stateSpaceFunctions)
    # import imp; import transformer.Translator; imp.reload(transformer.Translator)
    # import transformer.Beam; imp.reload(transformer.Beam)
    # import models.transformer_model; imp.reload(models.transformer_model)

    ## results of Beam search
    # can we add some stochasticity to beam search maybe?
    state_times, generated_sequences = model.generate(features, json_file, bpm, unique_states, generate_full_song=False)
    # state_times is the times of the nonemtpy states, in bpm units

    #%%
    from stateSpaceFunctions import stage_two_states_to_json_notes
    # times_real = [t*60/bpm for t in state_times]
    # times_real[]
    # np.arange(len(times_real))[:-1][np.diff(times_real) <= 0.07]
    # np.unique(np.diff(times_real), return_counts=True)
    # np.min(np.diff(times_real))
    # len(generated_sequences[0])
    # len(times_real)
    notes2 = stage_two_states_to_json_notes(generated_sequences[0], state_times, bpm, hop, sr, state_rank=unique_states)
    # notes2 = stage_two_states_to_json_notes(np.array(generated_sequences[0][:-2])[diff_mask].tolist(), times_filtered, bpm, hop, sr, state_rank=unique_states)

    # np.array(np.diff(times_real)) <= 0.1
    # np.diff(times_real)
    # np.all(np.isclose([t*bpm/60 for t in times_real], times))
    # np.unique(times_real, return_counts=True)
    # diff = np.diff(times_real)
    # diff1 = np.append(diff,10) <= 0.1
    # diff2 = np.insert(diff,0,10) <= 0.1
    # diff_mask = np.logical_or(diff1, diff2)
    # times_filtered = np.array(times_real)[diff_mask]
    #
    # np.diff(times) <= 0.125
    #
    len(notes2)
    # remake level with actual notes from stage 2 now
    make_level_from_notes(notes2, bpm, song_name, opt, args, open_in_browser=True)
