import pandas as pd
import numpy as np
import os
import librosa
import json
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import math
import numpy as np
from IOFunctions import parse_json
from glob import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
difficulties = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

def extract_note_times_from_notes(notes):
    return (o._time for o in notes)

def determine_minimum_block_interval(notes):
    note_times = extract_note_times_from_notes(notes)
    return np.min(note_times[1:-1] - note_times[0:-2])

if __name__ == '__main__':
    song_directory, song_ogg, song_json, song_filename = find_first_song_in_extract_directory()
    output_directory = song_directory + '_HMM_mod'

    for json in song_json.values():
        data = parse_json(json)
        #block_interval = determine_minimum_block_interval(['_notes'])
        #print('Minimum block interval for '+song_filename+': '+str(block_interval))
