import librosa
import numpy as np

from pathlib import Path

import pandas as pd

difficulty_levels = ["Easy","Normal","Hard","Expert","ExpertPlus"]

file_paths = pd.DataFrame(columns=['song']+difficulty_levels)

pathlist = Path("../oxai_beat_saber_data").glob('**/*.wav')
for path in pathlist:
    obj = {}
    path_in_str = str(path)
    # print(path_in_str)
    obj["song"]=path_in_str
    for level in difficulty_levels:
        level_path = "/".join(path.parts[:-1])+"/"+level+".json"
        if Path(level_path).exists():
            obj[level]=level_path
        else:
            obj[level]=None
    file_paths = file_paths.append(obj,ignore_index=True)

# file_paths

import json
# import math

def get_easiest_level(song):
    for level in difficulty_levels:
        if song[level] != None:
            return song[level]

import pandas as pd

# notes
# (notes['_time']/song_json['_beatsPerMinute'])*60

# SAMPLING_RATE = int(16e3)



# wave_file = file_paths.iloc[1]["song"]

# list(file_paths.iterrows())[0][1]["song"]

features_list = []
levels_list = []

for file_path in file_paths.iterrows():
    file_path = list(file_paths.iterrows())[0]
    print(file_path)
    file_path = file_path[1]

    wave_file = file_path["song"]
    wave_file = "../test_song.wav"

    if get_easiest_level(file_path) is None:
        print("no levels in this song?")
        continue
    song_json = json.load(open(get_easiest_level(file_path)))
    bpm = song_json['_beatsPerMinute']
    notes = pd.DataFrame(song_json['_notes'])

# song_json

    # load wave file
    wave, sr = librosa.load(wave_file, mono=True, sr=None)
    wave = wave[::3]

    sampling_rate = sr/3

    # bpm=librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(wave, sr=sr), sr=sr)[0]
    beat_duration = int(60*sampling_rate/bpm)

    # re-sample ( 48K -> 16K )

    BEAT_SUBDIVISION = 16

    mel_hop = beat_duration/BEAT_SUBDIVISION
    mel_window = 4*mel_hop
    # wave.shape[0]/mel_hop

    # get mfcc feature
    mfcc = librosa.feature.mfcc(wave, sr=sampling_rate, hop_length=mel_hop, n_fft=mel_window)

    # mfcc.shape

    # import matplotlib.pyplot as plt
    #
    # %matplotlib
    # import librosa.display
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # librosa.display.specshow(mfcc, x_axis='time')
    features_list.append(mfcc)

    blocks = np.zeros((mfcc.shape[1],4))

    # blocks[4]

    for note in notes.iterrows():
        note_time = int(note[1]['_time']*BEAT_SUBDIVISION)
        if note_time >=blocks.shape[0]:
            break
        blocks[note_time] = np.array([1+note[1]['_cutDirection'], note[1]['_lineIndex'], note[1]['_lineLayer'],note[1]['_type']])

    levels_list.append(blocks)
# plt.imshow(blocks,aspect='auto')

import pickle
pickle.dump(features_list,open("features_list.p","wb"))
# pickle.dump(features_list,open("test_features_list.p","wb"))
pickle.dump(levels_list,open("levels_list.p","wb"))
