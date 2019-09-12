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
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)

args = parser.parse_args()

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
song_path=args.song_path

from pathlib import Path
song_name = Path(song_path).stem

''' LOAD MODEL, OPTS, AND WEIGHTS'''
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
hop, features = extract_features(song_path, args, opt)

''' GENERATE LEVEL '''
#%%
song = torch.tensor(features).unsqueeze(0)

#generate level
# first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
first_samples = torch.full((1,opt.output_channels,receptive_field//2),constants.START_STATE)
print("Generating level timings... (sorry I'm a bit slow)")
if opt.concat_outputs: #whether to concatenate the generated outputs as new inputs (AUTOREGRESSIVE)
    output,peak_probs = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
    peak_probs = np.array(peak_probs)

    window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
    smoothed_peaks = np.convolve(peak_probs,window,mode='same')
    index = np.random.randint(len(smoothed_peaks))

    thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    print("number of peaks", len(peaks))
else: # NOT-AUTOREGRESSIVE (we keep it separate like this, because some models have both)
    # output = model.net.module.generate_no_autoregressive(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
    peak_probs = model.generate(features)[0,:,-1].cpu().detach().numpy()
    window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
    smoothed_peaks = np.convolve(peak_probs,window,mode='same')
    index = np.random.randint(len(smoothed_peaks))

    thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    print("number of peaks", len(peaks))
#%%

#convert from states to beatsaber notes
print("Processing notes...")
times_real = [float(i*hop/opt.sampling_rate) for i in peaks]
notes = [{"_time":float(t*args.bpm/60), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0} for t in times_real]
print("Number of generated notes: ", len(notes))
notes = np.array(notes)[np.where(np.diff([-1]+times_real) > constants.HUMAN_DELTA)[0]].tolist()

print("Number of generated notes (after pruning): ", len(notes))

json_file = make_level_from_notes(notes, args.bpm, song_name, args)

print(json_file)

#%%
