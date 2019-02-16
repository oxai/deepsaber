import librosa
import numpy as np

from pathlib import Path

import pandas as pd

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

BEAT_SUBDIVISION = 16
data_augmenting = False
beat_saber_data_path = "../oxai_beat_saber_data"

difficulty_levels = ["Easy","Normal","Hard","Expert","ExpertPlus"]

file_paths = pd.DataFrame(columns=['song']+difficulty_levels)

#load song paths
# pathlist = Path("../../oxai_beat_saber_data").glob('**/*.wav')
pathlist = Path(beat_saber_data_path).glob('**/*.ogg')

#test songs
# pathlist = ["../folder_where_test_wavs_are"]

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

# file_paths = list(file_paths.iterrows())

num_tasks = len(file_paths)

num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

# file_paths = comm.scatter(file_paths, root=0)

import json
# import math

def get_easiest_level(song):
    for level in difficulty_levels:
        if song[level] != None:
            return song[level]


def level_tensor_from_file(song_json,num_time_steps):
    notes = pd.DataFrame(song_json['_notes'])
    blocks = np.zeros((num_time_steps,4))

    for note in notes.iterrows():
        note_time = int(note[1]['_time']*BEAT_SUBDIVISION)
        if note_time >=blocks.shape[0]:
            break
        blocks[note_time] = np.array([1+note[1]['_cutDirection'], note[1]['_lineIndex'], note[1]['_lineLayer'],note[1]['_type']])

    return blocks

def add_noise(wave):
    noise = np.random.randn(len(wave))
    data_noise = wave + 0.005 * noise
    return data_noise

from librosa.effects import time_stretch, pitch_shift
# wave.shape
# # (wave, 1.1).shape
# stretch(wave,1.1).shape
# wave.shape
# pitch_shift(wave,sampling_rate,n_steps=1).shape

def mfcc_features(wave,sampling_rate,bpm):
    # load wave file

    # bpm=librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(wave, sr=sr), sr=sr)[0]
    beat_duration = int(60*sampling_rate/bpm) #beat duration in samples

    mel_hop = beat_duration//BEAT_SUBDIVISION #one vec of mfcc features per 16th of a beat (hop is in num of samples)
    mel_window = 4*mel_hop

    # get mfcc feature
    mfcc = librosa.feature.mfcc(wave, sr=sampling_rate, hop_length=mel_hop, n_fft=mel_window)
    return mfcc



#yeah global variables; and what?
features_list = []
levels_list = []

# list( file_paths.iterrows())[0][1]
def add_sample(wave,sampling_rate,bpm,song_json):
    mfcc = mfcc_features(wave,sampling_rate,bpm)
    features_list.append(mfcc)

    num_time_steps=mfcc.shape[1]
    levels_list.append(level_tensor_from_file(song_json,num_time_steps))

import time

for file_path in file_paths.iloc[tasks].iterrows():
    start = time. time()
    # file_path = list(file_paths.iterrows())[0]
    file_path = file_path[1]
    song_file_path = file_path["song"]
    print(song_file_path)
    if get_easiest_level(file_path) is None:
        print("no levels in this song?")
        continue

    wave_file = file_path["song"]
    wave, sampling_rate = librosa.load(wave_file, mono=True, sr=None)
    # re-sample ( 48K -> 16K ), I guess just to have less samples :P
    # wave = wave[::3]
    # sampling_rate = sr/3

    song_json = json.load(open(get_easiest_level(file_path)))
    bpm = song_json['_beatsPerMinute']

    add_sample(wave,sampling_rate,bpm,song_json)

    if data_augmenting:
        for ii in range(10):
            new_wave = add_noise(wave)
            librosa.output.write_wav(song_file_path+"_noise_"+str(ii)+".wav",new_wave,sampling_rate)
            add_sample(new_wave,sampling_rate, bpm,song_json)

        for rate in np.linspace(0.5,2,10):
            new_wave = time_stretch(wave,rate)
            librosa.output.write_wav(song_file_path+"_stretch_"+str(rate)+".wav",new_wave,sampling_rate)
            add_sample(new_wave,sampling_rate, rate*bpm,song_json)

        for shift in range(-3,3):
            new_wave = pitch_shift(wave,sampling_rate,n_steps=shift)
            librosa.output.write_wav(song_file_path+"_shift_"+str(shift)+".wav",new_wave,sampling_rate)
            add_sample(new_wave,sampling_rate,bpm,song_json)

    end = time. time()
    print(end-start)
    #the level array shouldn't really change, but I'm recalculating it because of possible discretization errors making the number of beats change upon time_stretching. Just in case..
    #for the adding noise, actually I wouldn't need to recalculate it. But that really isn't the bottleneck in speed, so not that important.

features_list = comm.gather(features_list, root=0)
levels_list = comm.gather(levels_list, root=0)
if rank == 0:
    features_list = sum(features_list,[])
    levels_list = sum(levels_list,[])
    import pickle
    pickle.dump(features_list,open("features_list.p","wb"))
    # pickle.dump(features_list,open("test_features_list.p","wb"))
    pickle.dump(levels_list,open("levels_list.p","wb"))
else:
    assert features_list is None
    assert levels_list is None


###if you wanna plot and listen


# import matplotlib.pyplot as plt
# #
# %matplotlib inline
#
# plt.plot(wave[:100])
# plt.plot(pitch_shift(wave,sampling_rate,n_steps=1)[:100])
#
# import IPython.display as ipd
# ipd.Audio(wave, rate=sampling_rate)
#
# ipd.Audio(pitch_shift(wave,sampling_rate,n_steps=5), rate=sampling_rate)
#

# notes

# import librosa.display
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# librosa.display.specshow(mfcc, x_axis='time')

# plt.imshow(blocks,aspect='auto')
