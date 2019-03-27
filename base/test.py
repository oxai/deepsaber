import sys
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber")
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model

sys.argv.append("--data_dir=../DataE/")
# sys.argv.append("--level_diff=Normal")
sys.argv.append("--batch_size=1")
sys.argv.append("--num_windows=5")
sys.argv.append("--gpu_ids=0")
#sys.argv.append("--nepoch=1")
#sys.argv.append("--nepoch_decay=1")
sys.argv.append("--output_length=95") # needs to be at least the receptive field (in time points) + 1 if using the GAN (adv_wavenet model)!
sys.argv.append("--layers=5")
sys.argv.append("--blocks=3")
sys.argv.append("--model=adv_wavenet")
sys.argv.append("--dataset_name=mfcc_look_ahead")
# sys.argv.append("--dataset_name=mfcc")
# sys.argv.append("--dataset_name=reduced_states")
# sys.argv.append("--experiment_name=mfcc_exp")
sys.argv.append("--experiment_name=gan_exp")
# sys.argv.append("--experiment_name=reduced_states_normal_exp")
# sys.argv.append("--experiment_name=reduced_states_exp")
#sys.argv.append("--print_freq=1")
#sys.argv.append("--workers=0")
#sys.argv.append("--output_length=1")


#these are useful for debugging/playing with Hydrogen@Atom, which Guille use
# sys.argv.pop(1)
# sys.argv.pop(1)

opt = TrainOptions().parse()
model = create_model(opt)
model.setup()

if opt.gpu_ids == -1:
    receptive_field = model.net.receptive_field
else:
    receptive_field = model.net.module.receptive_field

# model.load_networks('iter_34000')
# model.load_networks('iter_71000')
# model.load_networks('iter_22000')
# model.load_networks('iter_55000')
# model.load_networks('iter_18000')
model.load_networks('latest')

import librosa

# y, sr = librosa.load("../../test_song2.wav", sr=16000)
y, sr = librosa.load("../../test_song21.wav", sr=16000)
# y, sr = librosa.load("../../song2.ogg", sr=11025)

# bpm = 106 # 22
# bpm = 80 # 21
# bpm = 105 # 20
# bpm = 120 # 19
bpm = 128 # 18
# bpm = 76
# bpm=85 # 14
# bpm=91 #16
# bpm=67
# bpm=144
# bpm=100 #11
# bpm=106 # 6
# bpm=92
# bpm=166
# bpm = 97

beat_duration = int(60*sr/bpm) #beat duration in samples

# get mfcc feature
mel_hop = beat_duration//16
mel_window = 4*mel_hop
mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=20) #one vec of mfcc features per 16th of a beat (hop is in num of samples)

import torch

song = torch.tensor(mfcc).unsqueeze(0)

song.size(-1)

# output = model.net.module.generate(300,song, temperature=0.01)
output = model.net.module.generate(song.size(-1)-receptive_field,song,time_shifts=opt.time_shifts,temperature=1.0)

# receptive_field = model.net.module.receptive_field

# output[0,:,100]

# output

# output

import pickle
unique_states = pickle.load(open("../stateSpace/sorted_states2.pkl","rb"))

# unique_states

# list(enumerate(output[0,:,:].permute(1,0)))[-79][1]
#
# list(enumerate(list(enumerate(output[0,:,:].permute(1,0)))[100][1]))

states_list = output[0,:,:].permute(1,0)
# states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]

# states_list[0][0].int().item()

# tuple(12*[0])
#

# import numpy as  np
#
# [x for x in unique_states if x[3]==19 and x[2]==13]
#
# notes[40]

notes = [[{"_time":float(i/16.0), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
notes += [[{"_time":float(i/16.0), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
notes = sum(notes,[])

# 100//9 if 100//9 !=2 else 3

print(len(notes))

song_json = {u'_beatsPerBar': 16,
 u'_beatsPerMinute': bpm,
 u'_events': [],
 u'_noteJumpSpeed': 10,
 u'_notes': notes,
 u'_obstacles': [],
 u'_shuffle': 0,
 u'_shufflePeriod': 0.5,
 u'_version': u'1.5.0'}

import json

# with open("new_test_song14_reduced_states_temp1_0_55000.json", "w") as f:
# with open("new_test_song21_reduced_states_temp1_0_47000.json", "w") as f:
# with open("test_song18_new_mfcc_71000_temp1.json", "w") as f:
with open("test_song21_gan_latest1_temp1.json", "w") as f:
# with open("test_song18_new_mfcc_34000_Normal_temp1.json", "w") as f:
    f.write(json.dumps(song_json))


# import json
#
# level_json = json.load(open("Easy.json","r"))
#
# len(level_json["_notes"])
