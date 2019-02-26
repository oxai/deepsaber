import sys
sys.path.append("/home/guillefix/code/beatsaber/base")
sys.path.append("/home/guillefix/code/beatsaber")
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model

sys.argv.append("--data_dir=../../oxai_beat_saber_data/")
sys.argv.append("--dataset_name=mfcc")
sys.argv.append("--batch_size=1")
sys.argv.append("--num_windows=1")
sys.argv.append("--gpu_ids=0")
sys.argv.append("--nepoch=1")
sys.argv.append("--nepoch_decay=1")
sys.argv.append("--layers=5")
sys.argv.append("--blocks=3")
sys.argv.append("--print_freq=1")
sys.argv.append("--workers=0")
sys.argv.append("--output_length=1")


#these are useful for debugging/playing with Hydrogen@Atom, which Guille use
sys.argv.pop(1)
sys.argv.pop(1)

opt = TrainOptions().parse()
model = create_model(opt)
model.setup()

if opt.gpu_ids == -1:
    receptive_field = model.net.receptive_field
else:
    receptive_field = model.net.module.receptive_field

model.load_networks('latest')

import librosa

# y, sr = librosa.load("../../test_song.wav", sr=1600)
y, sr = librosa.load("../../song2.ogg", sr=1600)

bpm=125

beat_duration = int(60*sr/bpm) #beat duration in samples

# get mfcc feature
mel_hop = beat_duration//16
mel_window = 4*mel_hop
mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=mel_hop, n_fft=mel_window, n_mfcc=20) #one vec of mfcc features per 16th of a beat (hop is in num of samples)

import torch

song = torch.tensor(mfcc).unsqueeze(0)

song.size(-1)

output = model.net.module.generate(100,song, temperature=0.1)
# output = model.net.module.generate(song.size(-1)-receptive_field,song, temperature=0.1)

# output[0,:,100]

list(enumerate(output[0,:,:].permute(1,0)))[120][1]
#
# list(enumerate(list(enumerate(output[0,:,:].permute(1,0)))[100][1]))

notes = [[{"_time":float(i/16.0), "_cutDirection":int(y%9), "_lineIndex":int(j%4), "_lineLayer":int(j//4), "_type":(int(y//9) if int(y//9) !=2 else 3)} for j,y in enumerate(x) if y!=0] for i,x in enumerate(output[0,:,:].permute(1,0))]
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
 u'_shufflePeriod': 0.25,
 u'_version': u'1.5.0'}

import json

with open("Easy2.json", "w") as f:
    f.write(json.dumps(song_json))


# import json
#
# level_json = json.load(open("Easy.json","r"))
#
# len(level_json["_notes"])
