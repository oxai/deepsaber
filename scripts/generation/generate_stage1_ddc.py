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
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)
parser.add_argument('--ddc_file', type=str, default='test_ddc.sm')
parser.add_argument('--ddc_diff', type=int, default=1)

args = parser.parse_args()

temperature=args.temperature
song_path=args.song_path
ddc_file=args.ddc_file

from pathlib import Path
song_name = Path(song_path).stem

''' GENERATE LEVEL '''
#%%
diff = args.ddc_diff
print("Reading ddc file ", ddc_file)
notes = get_notes_from_stepmania_file(ddc_file, diff)
times_real = [(4*(60/125)/192)*note for note in notes]
#%%

#convert from states to beatsaber notes
print("Processing notes...")
notes = [{"_time":float(t*args.bpm/60), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0} for t in times_real]
print("Number of generated notes: ", len(notes))
notes = np.array(notes)[np.where(np.diff([-1]+times_real) > constants.HUMAN_DELTA)[0]].tolist()

print("Number of generated notes (after pruning): ", len(notes))

json_file = make_level_from_notes(notes, args.bpm, song_name, args)

print(json_file)

#%%
