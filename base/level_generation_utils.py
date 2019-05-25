
import json
generated_folder = "generated/"
logo_path = "logo.jpg"

def make_level_from_notes(notes, bpm, song_name, opt, args):
    temperature = args.temperature
    checkpoint = args.checkpoint


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

    signature_string = song_name+"_"+opt.model+"_"+opt.dataset_name+"_"+opt.experiment_name+"_"+str(temperature)+"_"+checkpoint
    json_file = generated_folder+"test_song"+signature_string+".json"
    with open(json_file, "w") as f:
        f.write(json.dumps(song_json))

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
    return json_file

#%%
