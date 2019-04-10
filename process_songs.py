
import numpy as np
import librosa
from pathlib import Path
import json
import os.path

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

data_path = Path("AugData")
#data_path = Path("DataE/")

candidate_audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())

num_tasks = len(candidate_audio_files)

num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

difficulty = "Expert"
sampling_rate = 16000
beat_subdivision = 16
n_mfcc = 20

receptive_field = 94
output_length = 95
input_length = receptive_field + output_length -1
time_shifts = 16

for i in tasks:
    path = candidate_audio_files[i]
    #print(path)
    song_file_path = path.__str__()
    mfcc_file = song_file_path+"_"+str(n_mfcc)+"_"+str(beat_subdivision)+"_mfcc.npy"
    try:
        #level = list(path.parent.glob(f'./{difficulty}.json'))[0]
        level = list(path.parent.glob('./'+difficulty+'.json'))[0]
        level = level.__str__()
    except (TypeError, IndexError):
            continue
    if not os.path.isfile(mfcc_file):
        #mfcc = pickle.load(open(mfcc_file,"rb"))
        #print("found mfcc file already")

        print("creating mfcc file",i)
        level = json.load(open(level, 'r'))

        bpm = level['_beatsPerMinute']
        sr = sampling_rate
        beat_duration = int(60*sr/bpm) #beat duration in samples

        mel_hop = beat_duration//beat_subdivision #one vec of mfcc features per 16th of a beat (hop is in num of samples)
        num_samples_per_feature = mel_hop + 1 
        mel_window = 4*mel_hop

        notes = level['_notes']

        # uncomment to look for notes beyond the end of time
        for note in notes:
            sample_index = receptive_field + floor((note['_time']*60/bpm)*sr/num_samples_per_feature)
            # check if note falls within the length of the song (why are there so many that don't??) #TODO: research why this happens
            if sample_index >= y.shape[1]:
                print("note beyond the end of time")
                continue

        # uncomment to create mfcc features
        #y, sr = librosa.load(song_file_path, sr=sampling_rate)

        ## get mfcc feature
        #mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=n_mfcc)


        #if mfcc.shape[1]-(input_length+time_shifts-1) < 1:
        #    print("Smol song, probably trolling; blacklisting...")
        #    with open(data_path.__str__()+"blacklist","a") as f:
        #        f.write(song_file_path+"\n")

        ##pickle.dump(mfcc,open(mfcc_file,"wb"))
        #np.save(mfcc_file,mfcc)
