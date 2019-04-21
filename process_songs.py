
import numpy as np
import librosa
from pathlib import Path
import json
import os.path
from stateSpaceFunctions import feature_extraction_hybrid_raw

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

# data_path = Path("AugData")
data_path = Path("DataE/")

candidate_audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())

num_tasks = len(candidate_audio_files)
# num_tasks = 10

num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

feature_name = "chroma"
feature_size = 24

difficulty = "Hard"
sampling_rate = 16000
beat_subdivision = 16
n_mfcc = 20

time_shifts = 16

# path = candidate_audio_files[4]

for i in tasks:
    path = candidate_audio_files[i]
    #print(path)
    song_file_path = path.__str__()
    features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+".npy"
    try:
        level = list(path.parent.glob('./'+difficulty+'.json'))[0]
        level = level.__str__()
    except (TypeError, IndexError):
            continue
    if not os.path.isfile(features_file):
        #mfcc = pickle.load(open(mfcc_file,"rb"))
        #print("found mfcc file already")

        print("creating feature file",i)
        level = json.load(open(level, 'r'))

        # get song
        y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)

        bpm = level['_beatsPerMinute']
        sr = sampling_rate
        beat_duration = int(60*sr/bpm) #beat duration in samples

        #hop = beat_duration//beat_subdivision #one vec of mfcc features per 16th of a beat (hop is in num of samples)
        hop = int(beat_duration * 1/beat_subdivision)
        hop -= hop % 32
        num_samples_per_feature = hop
        mel_window = hop

        notes = level['_notes']
        # from math import floor
        # uncomment to look for notes beyond the end of time
        # for note in notes:
        #     sample_index = floor((note['_time']*60/bpm)*sr/num_samples_per_feature)
        #     # check if note falls within the length of the song (why are there so many that don't??) #TODO: research why this happens
        #     if sample_index >= y_wav.shape[1]:
        #         print("note beyond the end of time")
        #         print((note['_time']*60/bpm)-(y_wav.shape[0]/sr))
        #         continue

        #get feature
        features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
        np.save(features_file,features)

        ## get mfcc feature
        # mfcc = librosa.feature.mfcc(y_wav, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=n_mfcc)
        # y = mfcc

# import matplotlib.pyplot as plt
# %matplotlib
# import IPython.display as ipd
#
# plt.matshow(features)
#
# sampling_rate = 22050
# y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)
# features = np.load(features_file)
# features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
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
