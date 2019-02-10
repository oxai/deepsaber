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
from decodeJSON import parse_json
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
    tracks = os.listdir(EXTRACT_DIR)
    searching = True
    i = 0
    while searching:
        candidate = os.path.join(EXTRACT_DIR, tracks[i])
        if os.path.isdir(candidate):
            inner_dir = os.path.join(candidate, os.listdir(candidate)[0])
            if os.path.isdir(inner_dir):
                candidate_ogg = glob(os.path.join(inner_dir, '*.ogg'))[0]
                if candidate_ogg:
                    searching = False
                    song_directory = inner_dir
                    song_ogg = candidate_ogg
        i+=1

    song_filename = song_ogg.split('/')[-1]
    song_json = dict()
    for difficulty in difficulties:
        json = os.path.join(song_directory, difficulty + '.json')
        if os.path.isfile(json):
            song_json[difficulty] = json
    output_directory = song_directory + '_HMM_mod'

    for json in song_json.values():
        data = parse_json(json)
        #block_interval = determine_minimum_block_interval(['_notes'])
        print('Minimum block interval for '+song_filename+': '+str(block_interval))
