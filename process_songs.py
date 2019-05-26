import numpy as np
import librosa
from pathlib import Path
import json
import os.path
from stateSpaceFunctions import feature_extraction_hybrid_raw, feature_extraction_mel,feature_extraction_hybrid

import sys
sys.argv[1]="AugData/"
sys.argv[2]="Expert"
data_path = Path(sys.argv[1])
#data_path = Path("DataE/")

# feature_name = "chroma"
feature_name = "mel"
feature_size = 100
# feature_size = 24
step_size = 0.01

replace_present=True
using_bpm_time_division = False

difficulties = sys.argv[2]
sampling_rate = 16000
beat_subdivision = 16
# n_mfcc = 20

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

candidate_audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
num_tasks = len(candidate_audio_files)
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

path = candidate_audio_files[0]

#%%

for i in tasks:
    path = candidate_audio_files[i]
    #print(path)
    song_file_path = path.__str__()
    features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+".npy"

    level_file_found = False
    for diff in difficulties.split(","):
        if Path(path.parent.__str__()+"/"+diff+".json").is_file():
            level = list(path.parent.glob('./'+diff+'.json'))[0]
            # level = list(path.parent.glob('./'+"Expert"+'.json'))[0]
            level = level.__str__()
            level_file_found = True
    if not level_file_found:
        continue

    if replace_present or not os.path.isfile(features_file):
        print("creating feature file",i)
        level = json.load(open(level, 'r'))

        # get song
        y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)

        bpm = level['_beatsPerMinute']
        sr = sampling_rate
        if using_bpm_time_division:
            beat_duration = 60/bpm #beat duration in seconds
            step_size = beat_duration/beat_subdivision #one vec of mfcc features per 16th of a beat (hop is in num of samples)
        else:
            beat_subdivision = 1/(step_size*bpm/60)

        #get feature
        #features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
        state_times = np.arange(0,y_wav.shape[0]/sr,step=step_size)
        if feature_name == "chroma":
            features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
        elif feature_name == "mel":
            # features = feature_extraction_hybrid(y_wav,sr,state_times,bpm,beat_subdivision=beat_subdivision,mel_dim=12)
            features = feature_extraction_mel(y_wav,sr,state_times,bpm,mel_dim=feature_size,beat_discretization=1/beat_subdivision)
            features = librosa.power_to_db(features, ref=np.max)
        np.save(features_file,features)
        # features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+"2.npy"
        # np.save(features_file,features)

        features.shape
        features = np.load(features_file)
        features.dtype

        # uncomment to look for notes beyond the end of time
        # notes = level['_notes']
        # from math import floor
        # for note in notes:
        #     sample_index = floor((note['_time']*60/bpm)*sr/num_samples_per_feature)
        #     # check if note falls within the length of the song (why are there so many that don't??) #TODO: research why this happens
        #     if sample_index >= y_wav.shape[1]:
        #         print("note beyond the end of time")
        #         print((note['_time']*60/bpm)-(y_wav.shape[0]/sr))
        #         continue

# import matplotlib.pyplot as plt
# %matplotlib
# import IPython.display as ipd
#
# plt.matshow(features)
#
# sampling_rate = 22050
# y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)
# features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
# features = np.load(features_file)
%matplotlib
import librosa.display
# features.shape[1]
# plt.matshow(features[:,:1000])
# plt.matshow(librosa.power_to_db(features, ref=np.max)[:,:100000])
librosa.display.specshow(features,x_axis='time')
# librosa.display.specshow(librosa.power_to_db(features, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# librosa.display.specshow(features[:12,:],x_axis='time')
# librosa.display.specshow(features[12:,:],x_axis='time')
# sample_index
# note['_time']
# sr*note['_time']*60/bpm
# sr*note['_time']*60/bpm/num_samples_per_feature
# print((note['_time']*60/bpm)-(y_wav.shape[0]/sr))
# y_wav.shape[0]
# y_wav.shape[0]/sr
#
# y_wav.shape[0]//mel_hop
#
# y.shape

# plt.plot(y)
# mfcc.shape
# plt.matshow(mfcc[:,-60:-1])
#
# sampling_rate = 16000
# y_harm, y_perc = librosa.effects.hpss(y_wav)
# ipd.Audio(y_perc, rate=sampling_rate)
# ipd.Audio(y_wav, rate=sampling_rate)
#
#
# ipd.Audio(y_harm, rate=sampling_rate)

# import pit

# ipd.Audio(pitch_shift(y,sampling_rate,n_steps=5), rate=sampling_rate)

# from process_beat_saber_data import pitch_shift

#if mfcc.shape[1]-(input_length+time_shifts-1) < 1:
#    print("Smol song, probably trolling; blacklisting...")
#    with open(data_path.__str__()+"blacklist","a") as f:
#        f.write(song_file_path+"\n")

##pickle.dump(mfcc,open(mfcc_file,"wb"))
#np.save(mfcc_file,mfcc)
