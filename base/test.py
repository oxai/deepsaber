import sys
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber")
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
import json
import librosa
import torch
import pickle
import os

#opt = TrainOptions().parse()
# open(opt.experiment_name+"/opt.json","w").write(json.dumps(vars(opt)))
experiment_name = "reduced_states_gan_exp_smoothedinput/"
opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

model = create_model(opt)
model.setup()
if opt.gpu_ids == -1:
    receptive_field = model.net.receptive_field
else:
    receptive_field = model.net.module.receptive_field

#%%

checkpoint = "2000"
checkpoint = "iter_"+checkpoint
# checkpoint = "latest"
model.load_networks(checkpoint)

#%%

# from pathlib import Path
song_number = "11"
song_name = "test_song"+song_number+".wav"
song_path = "../../"+song_name
y, sr = librosa.load(song_path, sr=16000)

bpms = {
"6": 106,
"11": 100,
"16": 91,
"14": 85,
"18": 128,
"19": 120,
"20": 105,
"21": 80,
"22": 106,
"24": 106,
"25": 66
}

bpm = bpms[song_number]
beat_duration = int(60*sr/bpm) #beat duration in samples

# get mfcc feature
mel_hop = beat_duration//16
mel_window = 4*mel_hop
mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=20) #one vec of mfcc features per 16th of a beat (hop is in num of samples)
song = torch.tensor(mfcc).unsqueeze(0)
song.size(-1)

#generate level
output = model.net.module.generate(song.size(-1)-receptive_field,song,time_shifts=opt.time_shifts,temperature=1.0)
states_list = output[0,:,:].permute(1,0)

#if using reduced_state representation convert from reduced_state_index to state tuple
unique_states = pickle.load(open("../stateSpace/sorted_states2.pkl","rb"))
states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]

#convert from states to beatsaber notes
notes = [[{"_time":float(i/16.0), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
notes += [[{"_time":float(i/16.0), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
notes = sum(notes,[])

print("Number of generated notes: ", len(notes))

#make song and info jsons
song_json = {u'_beatsPerBar': 16,
 u'_beatsPerMinute': bpm,
 u'_events': [],
 u'_noteJumpSpeed': 10,
 u'_notes': notes,
 u'_obstacles': [],
 u'_shuffle': 0,
 u'_shufflePeriod': 0.5,
 u'_version': u'1.5.0'}

info_json = {"songName":song_name,"songSubName":song_name,"authorName":"DeepSaber","beatsPerMinute":bpm,"previewStartTime":12,"previewDuration":10,"coverImagePath":"cover.jpg","environmentName":"NiceEnvironment","difficultyLevels":[{"difficulty":"Expert","difficultyRank":4,"audioPath":"song.ogg","jsonPath":"Expert.json"}]}

with open("test_song"+song_number+"_"+opt.model+"_"+opt.dataset_name+"_"+opt.experiment_name+"_"+checkpoint+".json", "w") as f:
    f.write(json.dumps(song_json))

generated_folder = "generated/"
logo_path = "logo.jpg"
level_folder = generated_folder+song_name+"/"+song_name
if not os.path.exists(generated_folder+song_name):
    os.makedirs(generated_folder+song_name)
if not os.path.exists(generated_folder+song_name+"/"+song_name):
    os.makedirs(level_folder )

with open(level_folder +"/Expert.json", "w") as f:
    f.write(json.dumps(song_json))

with open(level_folder +"/info.json", "w") as f:
    f.write(json.dumps(info_json))

from shutil import copyfile

copyfile(song_path, level_folder+"/song.ogg")
copyfile(logo_path, level_folder+"/cover.jpg")



#useful to inspect song..
# y.shape
# import matplotlib.pyplot as plt
# %matplotlib
# plt.plot(y)
# import IPython.display as ipd
# sampling_rate = 16000
# ipd.Audio(y, rate=sampling_rate)
#
# ipd.Audio(pitch_shift(y,sampling_rate,n_steps=5), rate=sampling_rate)
#
# from process_beat_saber_data import pitch_shift
