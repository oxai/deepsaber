import sys, os
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber")
sys.path.append("/home/guillefix/code/beatsaber/base/models")
import numpy as np
import librosa
import torch
from pathlib import Path
import json
import os.path
from process_scripts.feature_extraction.feature_extration import extract_features_hybrid, extract_features_mel,extract_features_hybrid_beat_synced
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
# %matplotlib

feature_name = "mel"
feature_size = 100
step_size = 0.01
sampling_rate = 16000
beat_subdivision = 16
using_bpm_time_division = False

data_path = "../AugData/"
diff = "Expert"
candidate_audio_files = sorted(Path(data_path).glob('**/*.ogg'), key=lambda path: path.parent.__str__())

#%%
'''LOADING EXAMPLE SONG'''

path = candidate_audio_files[0]
song_file_path = path.__str__()
# get song
y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)
# loading level
level_file = list(path.parent.glob('./'+diff+'.json'))[0]
level_file = level_file.__str__()
level = json.load(open(level_file, 'r'))
notes = level['_notes']

##time-relevant variables
bpm = level['_beatsPerMinute']
sr = sampling_rate
if using_bpm_time_division:
    beat_duration_samples = int(60*sr/bpm) #beat duration in samples
    hop = int(beat_duration_samples * 1/beat_subdivision)
    beat_duration = 60/bpm #beat duration in seconds
    step_size = beat_duration/beat_subdivision #in seconds
else:
    hop = int(step_size*sr)
    beat_subdivision = 1/(step_size*bpm/60)
l = features.shape[1]
sequence_length = l*step_size

#%%
##loading feature
features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+".npy"
features = np.load(features_file)
#%%

num_classes = 20
receptive_field = 1

blocks = np.zeros((l,12))
# reduced state version of the above. The reduced-state "class" at each time is represented as a one-hot vector of size `self.opt.num_classes`
blocks_reduced = np.zeros((l,num_classes))
# same as above but with class number, rather than one-hot, used as target
blocks_reduced_classes = np.zeros((l,1))

## CONSTRUCT BLOCKS TENSOR ##
for note in notes:
    #sample_index = floor((time of note in seconds)*sampling_rate/(num_samples_per_feature))
    #sample_index = floor((note['_time']*60/bpm)*sr/num_samples_per_feature)
    # we add receptive_field because we padded the y with 0s, to imitate generation
    sample_index = receptive_field + floor((note['_time']*60/bpm)*sr/hop - 0.5)
    # does librosa add some padding too?
    # check if note falls within the length of the song (why are there so many that don't??) #TODO: research why this happens
    if sample_index >= l:
        #print("note beyond the end of time")
        continue

    #constructing the representation of the block (as a number from 0 to 19)
    if note["_type"] == 3:
        note_representation = 19
    elif note["_type"] == 0 or note["_type"] == 1:
        note_representation = 1 + note["_type"]*9+note["_cutDirection"]
    else:
        raise ValueError("I thought there was no notes with _type different from 0,1,3. Ahem, what are those??")

    blocks[sample_index,note["_lineLayer"]*4+note["_lineIndex"]] = note_representation

#%%

## visualizing features
# # plt.matshow(features[:,:1000])
# # plt.matshow(librosa.power_to_db(features, ref=np.max)[:,:100000])
# librosa.display.specshow(features,x_axis='time')
# librosa.display.specshow(librosa.power_to_db(features, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# librosa.display.specshow(features[:12,:],x_axis='time')
# librosa.display.specshow(features[12:,:],x_axis='time')

# y_harm, y_perc = librosa.effects.hpss(y_wav)
# ipd.Audio(y_perc, rate=sampling_rate)
# ipd.Audio(y_wav, rate=sampling_rate)
# ipd.Audio(y_harm, rate=sampling_rate)

reading_notes = False
notes = []
index = 0

with open("test_ddc.sm", "r") as f:
    for line in f.readlines():
        line = line[:-1]
        if line=="#NOTES:":
            if not reading_notes:
                reading_notes = True
                continue
            else:
                break
        if line[0]!=" " and line[0]!=",":
            if reading_notes:
                if line!="0000":
                    # print(line)
                    notes.append(index)
                index += 1

event_times = [(60/125/192)*note for note in notes]
# notes
