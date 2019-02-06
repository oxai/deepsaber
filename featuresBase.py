import pandas as pd

# %%

import numpy as np
import os
import librosa
import json
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import math

def baseline_notes(ogg_file):
    # Load sample song
    # ong_path='C:\Users\gonca\Dropbox\oxai\beatsaber\believer\Believer\song.wav'
    # data, samplerate = sf.read('existing_file.wav')
    y, fs = librosa.load(ogg_file, sr=None)

    # define variables

    hop_length = 512
    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs)
    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
    # Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=fs)

    # %%
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=fs)
    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

    beat_chroma_chopped = beat_chroma[:, :-1]

    #beat_times_t = beat_times[:, None]
    #beat_times_T = beat_times_t.T

    # %% Convert beat times to time in beats

    beat_times_beats = beat_times * (125 / 60)

    # %% Convert beat chroma t into line Index and Layer
    indexMax = np.argmax(beat_chroma_chopped, axis=0)
    indexMax = indexMax + 1
    nCol = 4
    nRow = 3
    print(indexMax)

    line_layer = np.ones(indexMax.shape)
    line_index = np.ones(indexMax.shape)
    '''
    for i in range(0,len(indexMax)):
        line_layer[i] = math.ceil(indexMax[i] / nCol)

    line_index = indexMax - ((line_layer-1)*nCol)-1
    line_layer = line_layer - 1
    '''

    for i in range(0,len(indexMax)):
        line_layer[i] = math.floor((indexMax[i]-0.1) / 4)
    line_index = indexMax
    for i in range(0,len(indexMax)):
        line_index[i] = (indexMax[i]-1) % 4
    print(indexMax)
    #line_index = indexMax - ((line_layer-1)*nCol)-1
    #line_layer = line_layer - 1



    # %%
    #beat_features = np.vstack([beat_times_T, beat_chroma_chopped])

    # $$

    # %% Create Data frames

    dict = {'_cutDirection': [], '_lineIndex': [], '_lineLayer': [], '_time': [], '_type': [], }
    notes = pd.DataFrame.from_dict(dict)

    print(np.shape(beat_times_beats))

    for i in range(0,len(indexMax)):
        new_note = {'_cutDirection': [0], '_lineIndex': [line_index[i]], '_lineLayer': [line_layer[i]], '_time': [beat_times_beats[i]], '_type': [1]}
        df_new = pd.DataFrame.from_dict(new_note)
        notes = notes.append(df_new)

    return notes

def baseline_notes_simple():
    dict = {'_cutDirection' : [], '_lineIndex': [], '_lineLayer':[], '_time':[], '_type':[]}
    notes_random = pd.DataFrame.from_dict(dict)


    for i in range(0,418):
        new_note = {'_cutDirection' : [0], '_lineIndex':[1], '_lineLayer':[1], '_time':[i], '_type':[1]}
        df_new = pd.DataFrame.from_dict(new_note)
        notes_random = notes_random.append(df_new)
    return notes_random