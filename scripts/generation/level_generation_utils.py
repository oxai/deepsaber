import os
import json
generated_folder = "generated/"
logo_path = "logo.jpg"
import librosa
from scripts.feature_extraction.feature_extraction import extract_features_hybrid, extract_features_mel,extract_features_hybrid_beat_synced, extract_features_multi_mel

def extract_features(song_path, args, opt):
    y_wav, sr = librosa.load(song_path, sr=opt.sampling_rate)

    # useful quantities
    bpm = args.bpm
    feature_name = opt.feature_name
    feature_size = opt.feature_size
    sampling_rate = opt.sampling_rate
    beat_subdivision = opt.beat_subdivision
    try:
        step_size = opt.step_size
        using_bpm_time_division = opt.using_bpm_time_division
    except: # older model
        using_bpm_time_division = True

    sr = sampling_rate
    beat_duration = 60/bpm #beat duration in seconds
    beat_duration_samples = int(60*sr/bpm) #beat duration in samples
    if using_bpm_time_division:
        # duration of one time step in samples:
        hop = int(beat_duration_samples * 1/beat_subdivision)
        step_size = beat_duration/beat_subdivision # in seconds
    else:
        beat_subdivision = 1/(step_size*bpm/60)
        hop = int(step_size*sr)

    #get feature
    if feature_name == "chroma":
        if using_bpm_time_division:
            state_times = np.arange(0,y_wav.shape[0]/sr,step=step_size)
            features = extract_features_hybrid_beat_synced(y_wav,sr,state_times,bpm,beat_discretization=1/beat_subdivision)
        else:
            features = extract_features_hybrid(y_wav,sr,hop)
    elif feature_name == "mel":
        if using_bpm_time_division:
            raise NotImplementedError("Mel features with beat synced times not implemented, but trivial TODO")
        else:
            features = extract_features_mel(y_wav,sr,hop,mel_dim=feature_size)
    elif feature_name == "multi_mel":
        if using_bpm_time_division:
            raise NotImplementedError("Mel features with beat synced times not implemented, but trivial TODO")
        else:
            features = extract_features_multi_mel(y_wav, sr=sampling_rate, hop=hop, nffts=[1024,2048,4096], mel_dim=feature_size)

    return hop, features

def make_level_from_notes(notes, bpm, song_name, args, upload_to_dropbox=False, open_in_browser=False):
    temperature = args.temperature
    try:
        checkpoint = args.checkpoint
    except:
        checkpoint = ""
    song_path = args.song_path

    if open_in_browser:
        assert upload_to_dropbox

    #make song and info jsons
    song_json = {u'_events': [],
     u'_notes': notes,
     u'_obstacles': [],
     }

    info_json = {
      "_version": "2.0.0",
      "_songName": song_name,
      "_songSubName": "",
      "_songAuthorName": "DeepSaber",
      "_levelAuthorName": "DeepSaber",
      "_beatsPerMinute": bpm,
      "_songTimeOffset": 0,
      "_shuffle": 0,
      "_shufflePeriod": 0.5,
      "_previewStartTime": 12,
      "_previewDuration": 10,
      "_songFilename": "song.egg",
      "_coverImageFilename": "cover.jpg",
      "_environmentName": "NiceEnvironment",
      "_customData": {
        "_contributors": [],
        "_customEnvironment": "",
        "_customEnvironmentHash": ""
      },
      "_difficultyBeatmapSets": [
        {
          "_beatmapCharacteristicName": "Standard",
          "_difficultyBeatmaps": [
            {
              "_difficulty": "Expert",
              "_difficultyRank": 7,
              "_beatmapFilename": "Expert.dat",
              "_noteJumpMovementSpeed": 10,
              "_noteJumpStartBeatOffset": 0,
              "_customData": {
                "_difficultyLabel": "",
                "_editorOffset": 0,
                "_editorOldOffset": 0,
                "_warnings": [],
                "_information": [],
                "_suggestions": [],
                "_requirements": []
              }
            }
          ]
        }
      ]
    }

    try:
        try:
            signature = args.json_file.split("/")[1] + "_"
        except:
            signature = ""
        signature += args.experiment_name.replace("/","")+"_"+args.checkpoint
        try:
            signature +="_"+str(args.peak_threshold)
        except:
            pass
        try:
            if args.use_beam_search:
                signature += "_bs"
        except:
            pass
        try:
            signature +="_"+str(args.temperature)
        except:
            pass
    except:
        # signature = "ddc_".join([a+"_"+str(b).replace("/","") for a,b in vars(args).items()])
        signature = "ddc"+"_".join([a+"_"+str(b).replace("/","") for a,b in vars(args).items() if a !="ddc_file"])

    signature_string = song_name+"_"+signature
    json_file = generated_folder+signature_string+".dat"
    with open(json_file, "w") as f:
        f.write(json.dumps(song_json))

    level_folder = generated_folder+signature_string
    if not os.path.exists(level_folder):
        os.makedirs(level_folder)

    with open(level_folder +"/Expert.dat", "w") as f:
        f.write(json.dumps(song_json))

    with open(level_folder +"/info.dat", "w") as f:
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
        try:
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            return output
        except:
            print("couldn't run bash command, try running it manually")

    # bashCommand = "sox -t wav -b 16 "+song_path+" -t ogg "+ level_folder+"/song.ogg"
    bashCommand = "ffmpeg -y -i "+song_path+" -c:a libvorbis -q:a 4 "+ level_folder+"/song.ogg"
    run_bash_command(bashCommand)
    bashCommand = "mv "+level_folder+"/song.ogg"+" "+ level_folder+"/song.egg"
    run_bash_command(bashCommand)

    bashCommand = "zip -r "+generated_folder+song_name+"_"+signature_string+".zip "+level_folder
    run_bash_command(bashCommand)
    if open_in_browser:

        bashCommand = "./dropbox_uploader.sh upload "+generated_folder+song_name+"_"+signature_string+".zip /deepsaber_generated/"
        run_bash_command(bashCommand)

        bashCommand = "./dropbox_uploader.sh share /deepsaber_generated/"+song_name+"_"+signature_string+".zip"
        link = run_bash_command(bashCommand)
        demo_link = "https://supermedium.com/beatsaver-viewer/?zip=https://cors-anywhere.herokuapp.com/"+link[15:-2].decode("utf-8") +'1'
        print(demo_link)
        # run_bash_command("google-chrome "+demo_link)
    # zip -r test_song11 test_song11.wav
    # https://supermedium.com/beatsaver-viewer/?zip=https://cors-anywhere.herokuapp.com/https://www.dropbox.com/s/q67idk87u2f4rhf/test_song11.zip?dl=1
    # sox -t wav -b 16 ~/code/test_song11.wav -t ogg song.ogg
    return json_file

#%%

def get_notes_from_stepmania_file(ddc_file, diff):
    reading_notes = False
    index = 0
    counter = 0
    notes = []
    with open(ddc_file, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            if line=="#NOTES:":
                if counter == diff and not reading_notes:
                    reading_notes = True
                    counter += 1
                    continue
                elif counter > diff:
                    break
                else:
                    counter += 1
                    continue
            if reading_notes:
                if line[0]!=" " and line[0]!=",":
                    if line!="0000":
                        # print(line)
                        notes.append(index)
                    index += 1
    return notes
