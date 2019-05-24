#%%
import sys
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber/base/models")
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
import numpy as np
from transformer.Translator import Translator

from stateSpaceFunctions import feature_extraction_hybrid_raw,feature_extraction_mel,feature_extraction_hybrid

#opt = TrainOptions().parse()
# open(opt.experiment_name+"/opt.json","w").write(json.dumps(vars(opt)))

#%%
#experiment_name = "reduced_states_gan_exp_smoothedinput/"
# experiment_name = "reduced_states_lookahead_likelihood/"
# experiment_name = "zeropad_entropy_regularization/"
# experiment_name = "chroma_features_likelihood_exp1/"
# experiment_name = "chroma_features_likelihood_syncc/"
# experiment_name = "block_placement/"
experiment_name = "block_selection/"
# experiment_name = "transformer_testing/"
# experiment_name = "lstm_testing/"
# experiment_name = "chroma_features_likelihood_exp2/"
# experiment_name = "reduced_states_gan_exp_smoothedinput/"

opt = json.loads(open(experiment_name+"opt.json","r").read())
opt["gpu_ids"] = [0]
opt["cuda"] = True
opt["batch_size"] = 1
opt["beam_size"] = 5
opt["n_best"] = 5
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

# checkpoint = "64000"
checkpoint = "246000"
checkpoint = "iter_"+checkpoint
# checkpoint = "latest"
model.load_networks(checkpoint)
temperature=1.00


#%%

# from pathlib import Path
song_number = "43_fixed"
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
"21_fixed": 80,
"22": 106,
"23": 122,
"24": 106,
"24_fixed": 86,
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
"35_fixed": 128,
"36": 128,
"37": 129,
"38": 100,
"39": 130,
"40": 140,
"41": 72,
"42": 100,
"43": 118.5,
"43_fixed": 118.5,
"believer": 125,
}

bpm = bpms[song_number]

# get feature
feature_name = opt.feature_name
feature_size = opt.feature_size
use_sync=opt.using_sync_features

sampling_rate = opt.sampling_rate
beat_subdivision = opt.beat_subdivision
sr = sampling_rate
beat_duration = 60/bpm #beat duration in seconds

beat_duration_samples = int(60*sr/bpm) #beat duration in samples
# duration of one time step in samples:
hop = int(beat_duration_samples * 1/beat_subdivision)
if not use_sync:
    hop -= hop % 32
# num_samples_per_feature = hop

#
step_size = beat_duration/beat_subdivision #one vec of mfcc features per 16th of a beat (hop is in num of samples)
# mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=20) #one vec of mfcc features per 16th of a beat (hop is in num of samples)

state_times = np.arange(0,y_wav.shape[0]/sr,step=step_size)
if opt.feature_name == "chroma":
    if use_sync:
        features = feature_extraction_hybrid(y_wav,sr,state_times,bpm,beat_discretization=1/beat_subdivision,mel_dim=12)
    else:
        features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
elif opt.feature_name == "mel":
    assert use_sync
    # features = feature_extraction_hybrid(y_wav,sr,state_times,bpm,beat_subdivision=beat_subdivision,mel_dim=12)
    features = feature_extraction_mel(y_wav,sr,state_times,bpm,mel_dim=feature_size,beat_discretization=1/beat_subdivision)


# features = feature_extraction_hybrid_raw(y_wav,sr,bpm)
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

# output = model.net.module.generate(song)
# states_list = output[:,0,:]

#generate level
#output = model.net.module.generate(song.size(-1)-receptive_field,song,time_shifts=opt.time_shifts,temperature=1.0)
import Constants
first_samples = torch.full((1,opt.output_channels,receptive_field),Constants.START_STATE)
# first_samples = torch.full((1,opt.output_channels,receptive_field),Constants.EMPTY_STATE)
# first_samples[0,0,0] = Constants.START_STATE
if opt.concat_outputs:
    output = model.net.module.generate(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
else:
    output = model.net.module.generate_no_autoregressive(song.size(-1)-opt.time_shifts+1,song,time_shifts=opt.time_shifts,temperature=temperature,first_samples=first_samples)
states_list = output[0,:,:].permute(1,0)

#if using reduced_state representation convert from reduced_state_index to state tuple
unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
#old
# states_list = [(unique_states[i[0].int().item()-1] if i[0].int().item() != 0 else tuple(12*[0])) for i in states_list ]
#new (after transformer)

# notes

#convert from states to beatsaber notes
if opt.binarized:
    notes = [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_cutDirection":1, "_lineIndex":1, "_lineLayer":1, "_type":0}] for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
    # times_beat = [float((i+0.0)*bpm*hop/(sr*60)) for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
    times_real = [float((i+0.0)*hop/sr) for i,x in enumerate(states_list) if x[0].int().item() not in [0,1,2,3]]
else:
    states_list = [(unique_states[i[0].int().item()-4] if i[0].int().item() not in [0,1,2,3] else tuple(12*[0])) for i in states_list ]
    notes = [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_cutDirection":int((y-1)%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":int((y-1)//9)} for j,y in enumerate(x) if (y!=0 and y != 19)] for i,x in enumerate(states_list)]
    notes += [[{"_time":float((i+0.0)*bpm*hop/(sr*60)), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":3} for j,y in enumerate(x) if y==19] for i,x in enumerate(states_list)]
notes = sum(notes,[])

# song.size(-1)
# output.shape
# i*bpm*hop/(sr*60)
# y_wav.shape[0]/hop

#%%
# temperature=1.00

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

#%%

### STAGE TWO! ###


y = features
y = np.concatenate((np.zeros((y.shape[0],receptive_field)),y),1)
y = np.concatenate((y,np.zeros((y.shape[0],1))),1)
beat_duration = 60/bpm #beat duration in seconds
sample_duration = beat_duration * 1/opt.beat_subdivision #sample_duration in seconds
sequence_length = y.shape[1]*sample_duration

# json_file = "/home/guillefix/code/beatsaber/base/generated/test_song24_fixed_wavenet_general_beat_saber_block_placement_1.0_iter_58000.json"
generated_folder = "generated/"
# signature_string = song_number+"_"+opt.model+"_"+opt.dataset_name+"_"+opt.experiment_name+"_"+str(temperature)+"_"+checkpoint
level_folder = generated_folder+song_name
# level_folder="/home/guillefix/code/beatsaber/AugDataTest/4)Believer - Imagine Dragons/Believer"
json_file = level_folder +"/Expert.json"
# json_file = generated_folder+"test_song"+signature_string+".json"
## BLOCKS TENSORS ##
import imp; import stateSpaceFunctions; imp.reload(stateSpaceFunctions)
from stateSpaceFunctions import get_block_sequence_with_deltas
unique_states = pickle.load(open("../stateSpace/sorted_states.pkl","rb"))
one_hot_states, states, state_times, delta_forward, delta_backward, indices = get_block_sequence_with_deltas(json_file,sequence_length,bpm,top_k=2000,beat_discretization=1/opt.beat_subdivision,states=unique_states,one_hot=True,return_state_times=True)
truncated_sequence_length = min(len(states),opt.max_token_seq_len)
# truncated_sequence_length = min(len(states),512)
indices = indices[:truncated_sequence_length]
delta_forward = delta_forward[:,:truncated_sequence_length]
delta_backward = delta_backward[:,:truncated_sequence_length]
# pos_enc = pos_enc[:truncated_sequence_length]

# input_block_sequence = torch.tensor(one_hot_states).unsqueeze(0).long()
input_forward_deltas = torch.tensor(delta_forward).unsqueeze(0).long()
input_backward_deltas = torch.tensor(delta_backward).unsqueeze(0).long()

y = y[:,indices]
input_windows = [y]

song_sequence = torch.tensor(input_windows)
song_sequence = (song_sequence - song_sequence.mean())/torch.abs(song_sequence).max().float()
song_sequence = torch.cat([song_sequence,input_forward_deltas.double(),input_backward_deltas.double()],1)

src_pos = torch.tensor(np.arange(len(indices))).unsqueeze(0)
src_mask = torch.tensor(4*np.ones(len(indices))).unsqueeze(0)

song_sequence.shape
src_pos.shape
src_mask.shape

#%%

# model.net.module.encoder(song_sequence.permute(0,2,1).float().cuda(),src_mask.cuda(),src_pos.cuda())

import imp; import transformer.Translator; imp.reload(transformer.Translator)
import transformer.Beam; imp.reload(transformer.Beam)
from transformer.Translator import Translator

translator = Translator(opt,model)
all_hyp, all_scores = translator.translate_batch(song_sequence.permute(0,2,1).float(), src_pos, src_mask,truncated_sequence_length)

# need to pass to beam .advance, the length of sequence :P ... I think it makes sense

len(all_hyp[0])
all_hyp[0]
all_hyp[0][0]
len(all_hyp[0][0])
all_scores

#%%

# unique_states[60-3]

from stateSpaceFunctions import stage_two_states_to_json_notes

notes = stage_two_states_to_json_notes(all_hyp[0][1], state_times, bpm, hop, sr, state_rank=unique_states)
notes

# opt.max_token_seq_len

#%%

# features.shape
#
# #useful to inspect song..
# # y.shape
# import matplotlib.pyplot as plt
# %matplotlib
# plt.plot(y_wav)
# import numpy as np
# D = np.abs(librosa.stft(y_wav))
# D.shape
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time',sr=sr)
#
# mfcc=librosa.feature.mfcc(y_wav,sr=sr)
# import librosa.display
# librosa.display.specshow(mfcc, x_axis='time',sr=sr)
# chroma=librosa.feature.chroma_cqt(y_wav,sr=sr)
# chroma=librosa.feature.chroma_stft(y_wav,sr=sr)
# chroma=librosa.feature.chroma_cens(y_wav,sr=sr)
# librosa.display.specshow(chroma, x_axis='time',sr=sr)
#
# D = np.abs(librosa.stft(y_wav))**2
# S=librosa.feature.melspectrogram(S=D,sr=sr)
# S.shape
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time',sr=sr)
# librosa.display.specshow(melspec, x_axis='time',sr=sr)
#
# # librosa.display.specshow(features[12:], x_axis='time')
# # librosa.display.specshow(features[:12], x_axis='time')
# import IPython.display as ipd
# # sampling_rate = 16000
# ipd.Audio(y_wav, rate=sr)
# #
# # ipd.Audio(pitch_shift(y,sampling_rate,n_steps=5), rate=sampling_rate)
# #
# # from process_beat_saber_data import pitch_shift
