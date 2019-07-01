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
from math import ceil

from stateSpaceFunctions import feature_extraction_hybrid_raw,feature_extraction_mel,feature_extraction_hybrid

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
# class Struct:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
# args = Struct(**args)

# signature = "_".join([a+"_"+str(b).replace("/","") for a,b in vars(args).items()])

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
song_path=args.song_path
ddc_file=args.ddc_file
if args.two_stage:
    assert args.experiment_name2 is not None
    assert args.checkpoint2 is not None

# song_name = "18"
# song_name = "test_song"+song_name+".wav"
# song_path = song_dir+song_name
# song_path = "../../"+song_name
from pathlib import Path
song_name = Path(song_path).stem
# print(experiment_name)

''' LOAD MODEL, OPTS, AND WEIGHTS (for stage1 if two_stage) '''
#%%

#loading opt object from experiment
opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
opt["cuda"] = True
opt["experiment_name"] = args.experiment_name.split("/")[0]
if "dropout" not in opt:
    opt["dropout"] = 0.0
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
if not args.use_ddc:
    song = torch.tensor(features).unsqueeze(0)
    # temperature = 1.00

    #generate level
    first_samples = torch.full((1,opt.output_channels,receptive_field//2),Constants.START_STATE)
    # first_samples = torch.full((1,receptive_field//2),Constants.START_STATE)
    # stuff for older models (will remove at some point:)
    # first_samples = torch.full((1,opt.output_channels,receptive_field),Constants.EMPTY_STATE)
    # first_samples[0,0,0] = Constants.START_STATE
    print("Generating level timings... (sorry I'm a bit slow)")
    if opt.concat_outputs:
        output,peak_probs = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
    else:
        output = model.net.module.generate_no_autoregressive(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
    states_list = output[0,:,:].permute(1,0)
    # states_list.tolist()
    # np.unique(states_list)

    peak_probs = np.array(peak_probs)

    import matplotlib.pyplot as plt
    # plt.plot(peak_probs[0:300])

    from scipy import signal
    # window = signal.hamming(15)
    window = signal.hamming(ceil(Constants.HUMAN_DELTA/opt.step_size))

    smoothed_peaks = np.convolve(peak_probs,window,mode='same')
    index = np.random.randint(len(smoothed_peaks))
    plt.plot(smoothed_peaks[index:index+1000])

    plt.savefig("smoothed_peaks.png")
    pickle.dump(smoothed_peaks, open("smoothed_peaks.p","wb"))

    # thresholded_peaks = smoothed_peaks*(smoothed_peaks>0.0148)
    thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
    # thresholded_peaks = smoothed_peaks*(smoothed_peaks>0.46)

    # plt.plot(thresholded_peaks[:])

    peaks = signal.find_peaks(thresholded_peaks)[0]

    print("number of peaks", len(peaks))

#%%

if args.use_ddc:

    reading_notes = False
    notes = []
    index = 0
    diff = args.ddc_diff
    counter = 0
    print("Reading ddc file ", ddc_file)
    with open(ddc_file, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            if line=="#NOTES:":
                if counter == diff and not reading_notes:
                    reading_notes = True
                    counter += 1
                    continue
                elif counter > diff:
                    break
                else:
                    counter += 1
                    continue
            if reading_notes:
                if line[0]!=" " and line[0]!=",":
                    if line!="0000":
                        # print(line)
                        notes.append(index)
                    index += 1

    # len(times_real)
    times_real = [(4*(60/125)/192)*note for note in notes]
    # notes

#%%


#if using reduced_state representation convert from reduced_state_index to state tuple
#old (before transformer)
# states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]

#convert from states to beatsaber notes
print("Processing notes...")
if opt.binarized: # for experiments where the output is state/no state
    if not args.use_ddc:
        times_real = [float(i*hop/sr) for i in peaks]
    notes = [{"_time":float(t*bpm/60), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0} for t in times_real]
    print("Number of generated notes: ", len(notes))
    notes = np.array(notes)[np.where(np.diff([-1]+times_real) > Constants.HUMAN_DELTA)[0]].tolist()
else: # this is where the notes are generated for end-to-end models that actually output states
    unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
    states_list = [(unique_states[i[0].int().item()-4] if i[0].int().item() not in [0,1,2,3] else tuple(12*[0])) for i in states_list ]
    notes = [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
    notes += [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
    notes = sum(notes,[])

print("Number of generated notes (after pruning): ", len(notes))

json_file = make_level_from_notes(notes, bpm, song_name, opt, args)
# json_file = make_level_from_notes(notes, bpm, song_name, opt, args, upload_to_dropbox=True, open_in_browser=True)
# notes
# list(map(lambda x: ))
# times = [note["_time"] for note in notes]
# np.unique(np.diff(times_real), return_counts=True)
# np.diff(times_real) <= 0.125
# np.diff(times) <= 0.125
# len(times) = len()


#%%

''' STAGE TWO! '''

if args.two_stage:
    print("STAGE TWO!")
    #%%
    ''' LOAD MODEL, OPTS, AND WEIGHTS (for stage2 if two_stage) '''
    experiment_name = args.experiment_name2+"/"
    # experiment_name = "block_selection_new/"
    checkpoint = args.checkpoint2
    # checkpoint = "975000"
    # checkpoint = "1080000"

    #loading opt object from experiment
    opt = json.loads(open(experiment_name+"opt.json","r").read())
    # extra things Beam search wants
    opt["gpu_ids"] = [0]
    opt["cuda"] = True
    opt["batch_size"] = 1
    opt["beam_size"] = 20
    opt["n_best"] = 1
    opt["using_bpm_time_division"] = True
    opt["continue_train"] = False
    # opt["max_token_seq_len"] = len(notes)
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
    # json_file = 'generated/test_songtest_song35_fixed.wav_wavenet_general_beat_saber_block_placement_new_1.0_60000.json'
    # json_file = 'generated/test_songtest_song35_fixed.wav_wavenet_general_beat_saber_block_placement_new_nohumreg_1.0_65000.json'

    #%%
    # import imp; import stateSpaceFunctions; imp.reload(stateSpaceFunctions)
    # import imp; import transformer.Translator; imp.reload(transformer.Translator)
    # import transformer.Beam; imp.reload(transformer.Beam)
    # import models.transformer_model; imp.reload(models.transformer_model)

    ## results of Beam search
    # can we add some stochasticity to beam search maybe?
    # state_times, generated_sequences = model.generate(features, json_file, bpm, unique_states, generate_full_song=False)
    # len(state_times)
    # state_times, generated_sequence = model.generate(features, json_file, bpm, unique_states, generate_full_song=False)
    print("Generating state sequence...")
    state_times, generated_sequence = model.generate(features, json_file, bpm, unique_states, temperature=temperature, use_beam_search=args.use_beam_search, generate_full_song=args.generate_full_song)
    # state_times is the times of the nonemtpy states, in bpm units

    #%%
    from stateSpaceFunctions import stage_two_states_to_json_notes
    times_real = [t*60/bpm for t in state_times]
    # times_real[]
    # np.arange(len(times_real))[:-1][np.diff(times_real) <= 0.07]
    # np.unique(np.diff(times_real), return_counts=True)
    # np.min(np.diff(times_real))
    # len(generated_sequences[0])
    # len(times_real)
    notes2 = stage_two_states_to_json_notes(generated_sequence, state_times, bpm, hop, sr, state_rank=unique_states)
    # notes2 = stage_two_states_to_json_notes(generated_sequences[0], state_times, bpm, hop, sr, state_rank=unique_states)
    # notes2 = stage_two_states_to_json_notes(np.array(generated_sequences[0][:-2])[diff_mask].tolist(), times_filtered, bpm, hop, sr, state_rank=unique_states)

    # np.array(np.diff(times_real)) <= 0.1
    # np.diff(times_real)
    # np.all(np.isclose([t*bpm/60 for t in times_real], times))
    print("Bad notes:", np.unique(np.diff(times_real)[np.diff(times_real)<=Constants.HUMAN_DELTA], return_counts=True))
    # diff = np.diff(times_real)
    # diff1 = np.append(diff,10) <= 0.1
    # diff2 = np.insert(diff,0,10) <= 0.1
    # diff_mask = np.logical_or(diff1, diff2)
    # times_filtered = np.array(times_real)[diff_mask]
    #
    # np.diff(times) <= 0.125
    #
    print(len(notes2))
    # remake level with actual notes from stage 2 now
    print("Generating level...")
    print("Uploading to dropbox...")
    make_level_from_notes(notes2, bpm, song_name, opt, args, upload_to_dropbox=True, open_in_browser=args.open_in_browser)
