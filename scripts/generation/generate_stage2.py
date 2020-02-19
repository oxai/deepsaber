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
parser.add_argument('--json_file', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)
parser.add_argument('--generate_full_song', action="store_true")
parser.add_argument('--use_beam_search', action="store_true")
parser.add_argument('--open_in_browser', action="store_true")
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
song_path=args.song_path
json_file=args.json_file

from pathlib import Path
song_name = Path(song_path).stem

print("STAGE TWO!")
#%%
''' LOAD MODEL, OPTS, AND WEIGHTS (for stage1 if two_stage) '''

#loading opt object from experiment, and constructing Struct object after adding some things
opt = json.loads(open("../training/"+experiment_name+"opt.json","r").read())
# extra things Beam search wants
if args.cuda:
    opt["gpu_ids"] = [0]
else:
    opt["gpu_ids"] = []
opt["load_iter"] = int(checkpoint)
if args.cuda:
    opt["cuda"] = True
else:
    opt["cuda"] = False
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

make_level_from_notes(notes2, args.bpm, song_name, args, upload_to_dropbox=args.open_in_browser, open_in_browser=args.open_in_browser, copy_to_root=True)
