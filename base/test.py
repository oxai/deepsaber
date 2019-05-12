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

from stateSpaceFunctions import feature_extraction_hybrid_raw

#opt = TrainOptions().parse()
# open(opt.experiment_name+"/opt.json","w").write(json.dumps(vars(opt)))

#%%
#experiment_name = "reduced_states_gan_exp_smoothedinput/"
# experiment_name = "reduced_states_lookahead_likelihood/"
# experiment_name = "zeropad_entropy_regularization/"
# experiment_name = "chroma_features_likelihood_exp1/"
experiment_name = "lstm_testing/"
# experiment_name = "chroma_features_likelihood_exp2/"
# experiment_name = "reduced_states_gan_exp_smoothedinput/"

opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

model = create_model(opt)
model.setup()
if opt.model=='wavenet' or opt.model=='adv_wavenet':
    if not opt.gpu_ids:
        receptive_field = model.net.receptive_field
    else:
        receptive_field = model.net.module.receptive_field
else:
    receptive_field = 1

#%%

# checkpoint = "590000"
checkpoint = "21000"
checkpoint = "iter_"+checkpoint
# checkpoint = "latest"
model.load_networks(checkpoint)

#%%

# from pathlib import Path
song_number = "16"
print("Song number: ",song_number)
song_name = "test_song"+song_number+".wav"
song_path = "../../"+song_name
y_wav, sr = librosa.load(song_path, sr=16000)

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
"23": 122,
"24": 106,
"25": 66,
"26": 68,
"27": 84,
"28": 114,
"29": 48,
"30": 108,
"31": 80,
"32": 52,
"33": 60,
"34": 80,
# "35": 256,
"35": 128,
"36": 130,
"37": 129,
"38": 100,
"39": 130,
"40": 140,
"41": 72,
"42": 100,
}

bpm = bpms[song_number]

# get mfcc feature
beat_duration = int(60 * sr / bpm)  # beat duration in samples
hop = int(beat_duration * (1/16)) # one vec of mfcc features per 16th of a beat (hop is in num of samples)
hop -= hop % 32
mel_window = 1*hop
# mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=20) #one vec of mfcc features per 16th of a beat (hop is in num of samples)

features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
# import matplotlib.pyplot as plt
# %matplotlib
# import IPython.display as ipd
# import librosa.display
# librosa.display.specshow(features[:12,:],x_axis='time')
# librosa.display.specshow(features[12:,:],x_axis='time')
# ipd.Audio(y_wav, rate=sr)


song = torch.tensor(features).unsqueeze(0)
# song.size(-1)

#%%

temperature=1.00
output = model.net.module.generate(song)
states_list = output[:,0,:]

#generate level
#output = model.net.module.generate(song.size(-1)-receptive_field,song,time_shifts=opt.time_shifts,temperature=1.0)
# output = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature)
# states_list = output[0,:,:].permute(1,0)

#if using reduced_state representation convert from reduced_state_index to state tuple
unique_states = pickle.load(open("../stateSpace/sorted_states2.pkl","rb"))
states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]

#convert from states to beatsaber notes
notes = [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
notes += [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
notes = sum(notes,[])

# song.size(-1)
# output.shape
# i*bpm*hop/(sr*60)
# y_wav.shape[0]/hop


print("Number of generated notes: ", len(notes))

#make song and info jsons
song_json = {u'_beatsPerBar': 4,
 u'_beatsPerMinute': bpm,
 u'_events': [],
 u'_noteJumpSpeed': 10,
 u'_notes': notes,
 u'_obstacles': [],
 u'_shuffle': 0,
 u'_shufflePeriod': 0.5,
 u'_version': u'1.5.0'}

info_json = {"songName":song_name,"songSubName":song_name,"authorName":"DeepSaber","beatsPerMinute":bpm,"previewStartTime":12,"previewDuration":10,"coverImagePath":"cover.jpg","environmentName":"NiceEnvironment","difficultyLevels":[{"difficulty":"Expert","difficultyRank":4,"audioPath":"song.ogg","jsonPath":"Expert.json"}]}

generated_folder = "generated/"
signature_string = song_number+"_"+opt.model+"_"+opt.dataset_name+"_"+opt.experiment_name+"_"+str(temperature)+"_"+checkpoint
with open(generated_folder+"test_song"+signature_string+".json", "w") as f:
    f.write(json.dumps(song_json))

logo_path = "logo.jpg"
level_folder = generated_folder+song_name
if not os.path.exists(level_folder):
    os.makedirs(level_folder)

with open(level_folder +"/Expert.json", "w") as f:
    f.write(json.dumps(song_json))

with open(level_folder +"/info.json", "w") as f:
    f.write(json.dumps(info_json))

from shutil import copyfile

copyfile(logo_path, level_folder+"/cover.jpg")
# copyfile(song_path, level_folder+"/song.ogg")

#import soundfile as sf
# y, sr = librosa.load(song_path, sr=48000)
# sf.write(level_folder+"/song.ogg", y, sr, format='ogg', subtype='vorbis')

import subprocess
def run_bash_command(bashCommand):
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

bashCommand = "sox -t wav -b 16 "+song_path+" -t ogg "+ level_folder+"/song.ogg"
run_bash_command(bashCommand)

bashCommand = "zip -r "+generated_folder+song_name+"_"+signature_string+".zip "+level_folder
run_bash_command(bashCommand)

bashCommand = "./dropbox_uploader.sh upload "+generated_folder+song_name+"_"+signature_string+".zip /deepsaber_generated/"
run_bash_command(bashCommand)

bashCommand = "./dropbox_uploader.sh share /deepsaber_generated/"+song_name+"_"+signature_string+".zip"
link = run_bash_command(bashCommand)
demo_link = "https://supermedium.com/beatsaver-viewer/?zip=https://cors-anywhere.herokuapp.com/"+link[15:-2].decode("utf-8") +'1'
print(demo_link)
run_bash_command("google-chrome "+demo_link)
# zip -r test_song11 test_song11.wav
# https://supermedium.com/beatsaver-viewer/?zip=https://cors-anywhere.herokuapp.com/https://www.dropbox.com/s/q67idk87u2f4rhf/test_song11.zip?dl=1
# sox -t wav -b 16 ~/code/test_song11.wav -t ogg song.ogg

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
