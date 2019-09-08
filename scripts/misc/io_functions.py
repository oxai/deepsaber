import json, re
import pandas as pd
import os
import time
import getpass
import pickle
from glob import glob
import html

from matplotlib import pyplot

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

difficulties = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

def saveFile(object, filename=None, save_dir=None, append=False):
    if save_dir is None or save_dir is '':
        save_dir = os.path.join(os.getcwd(), 'Temp')
    if not os.path.isdir(save_dir):  # SUBJECT TO RACE CONDITION
        os.mkdir(save_dir)
    if filename is None or filename is '':
        filename = os.path.join(save_dir,
                                getpass.getuser() + '_' + time.strftime('%Y-%m-%d_%H:%M:%S') + '.tmp')
    else:
        filename = os.path.join(save_dir, filename)
    print('Saving file: ' + filename)
    if append is True:
        savefile = open(filename, 'ab')
    else:
        savefile = open(filename, 'wb')
    pickle.dump(object, savefile)
    savefile.close()
    return filename


def loadFile(filename, load_dir=None):
    if load_dir is None or load_dir is '':
        load_dir = os.path.join(os.getcwd(), 'Temp')
    if not os.path.isdir(load_dir):  # SUBJECT TO RACE CONDITION
        print("Directory does not exist, creating directory.")
        os.mkdir(load_dir)
    filename = os.path.join(load_dir, filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def parse_json(json_filepath):
    # input json file
    # returns: dataframes of: events, notes, obstacles, as well as additional parameters (in the form of a dictionary)
    json_data = open(json_filepath).read()
    data = json.loads(json_data)
    events = data.get('_events')
    notes = data.get('_notes')
    obstacles = data.get('_obstacles')

    #create the dataframes
    df_events = pd.DataFrame(events)
    df_notes = pd.DataFrame(notes)
    df_obstacles = pd.DataFrame(obstacles)

    #update dataframes in the data file
    data['_events'] = df_events
    data['_notes'] = df_notes
    data['_obstacles'] = df_obstacles
    return data


def get_song_from_directory_by_identifier(identifier, difficulty=None):
    song_directory = os.path.join(EXTRACT_DIR, identifier)
    if len(os.listdir(song_directory)) == 1 and os.path.isdir(os.listdir(song_directory)[0]):
        song_directory = os.path.join(song_directory, os.listdir(song_directory)[0])
    song_json = dict()
    if difficulty is None:
        for difficulty in difficulties:
            json = os.path.join(song_directory, difficulty + '.dat')
            if os.path.isfile(json):
                song_json[difficulty] = json
    else:
        json = os.path.join(song_directory, difficulty + '.dat')
        if os.path.isfile(json):
            song_json = json
        else:
            raise Exception('No file of difficulty: '+difficulty+' in: '+song_directory)
    song_ogg = glob(os.path.join(song_directory, '*.egg'))[0]
    song_filename = song_ogg.split('/')[-1]
    return song_directory, song_ogg, song_json, song_filename


def get_first_song_in_extract_directory():
    tracks = os.listdir(EXTRACT_DIR)
    searching = True
    i = 0
    while searching:
        candidate = os.path.join(EXTRACT_DIR, tracks[i])
        if os.path.isdir(candidate):
            inner_dir = os.path.join(candidate, os.listdir(candidate)[0])
            if os.path.isdir(inner_dir):
                candidate_ogg = glob(os.path.join(inner_dir, '*.ogg'))[0]
                if candidate_ogg:
                    searching = False
                    song_directory = inner_dir
                    song_ogg = candidate_ogg
        i += 1
    song_filename = song_ogg.split('/')[-1]
    song_json = dict()
    for difficulty in difficulties:
        json = os.path.join(song_directory, difficulty + '.json')
        if os.path.isfile(json):
            song_json[difficulty] = json
    return song_directory, song_ogg, song_json, song_filename


def make_integers(list_of_dict):
    # converts the dictionary entries to integers - where possible --> necessary for visualizer compatibility
    for item in list_of_dict:
        for key, value in item.items():
            try:
                if key == '_time':
                    item[key] = float(value)
                else:
                    item[key] = int(value)
            except ValueError:
                item[key] = float(value)
    return list_of_dict


def create_dataStructure(df_events, df_notes, df_obstacles, version, shufflePeriod, noteJumpSpeed, beatsPerBar, shuffle, bpm):
    #takes in the data information and returns a dictionary that can be encoded to json via encode_json()

    #events = dict['_events'] #df
    #notes = dict['_notes'] #df
    #obstacles = dict['_obstacles'] #df
    ##additional information
    #version = dict['_version'] # string
    #shufflePeriod = dict['_shufflePeriod'] #int
    #noteJumpSpeed = dict['_noteJumpSpeed'] #int
    #beatsPerBar = dict['_beatsPerBar'] #int
    #shuffle = dict['_shuffle'] #int
    #bpm = dict['_beatsPerMinute'] #int

    #convert datframe to list of dicts
    events = df_events.to_dict('records')
    notes = df_notes.to_dict('records')
    obstacles = df_obstacles.to_dict('records')

    #make them to integer values
    events = make_integers(events)
    notes = make_integers(notes)
    obstacles = make_integers(obstacles)

    #output the dictionary in the right format
    dict = {'_obstacles': obstacles, '_beatsPerMinute': bpm, '_noteJumpSpeed': noteJumpSpeed, '_version': version, '_events': events, '_shuffle': shuffle, '_shufflePeriod': shufflePeriod, '_beatsPerBar': beatsPerBar, '_notes': notes}
    return dict


def encode_json(data, file_directory):
    #encode the dictionary
    with open(file_directory, 'w') as fp:
        json.dump(data, fp)

    return 0


'''@RA: There is some redundancy in the getting OGG and JSON. I recommend switching to the below two functions, ,
which are optimised. (Technically these two fcts can be made one with an extra param, but this is simpler)'''


def get_all_ogg_files_from_data_directory(data_directory):
    file_regex = re.compile("^.*\\.egg$")
    ogg_files = []
    for root, subdirectories, files in os.walk(data_directory):
        for name in files:
            if bool(file_regex.match(name)):
                ogg_path = os.path.join(root, name)
                ogg_files.append(ogg_path)
    return ogg_files


def get_all_json_level_files_from_data_directory(data_directory, include_autosaves=False):
    '''@RA: Identify All JSON files in our data set.
    Ref: https://stackoverflow.com/questions/2909975/python-list-directory-subdirectory-and-files '''
    if include_autosaves:
        file_regex = re.compile("^.*(Easy|Normal|Hard|Expert|ExpertPlus)\\.json$")
    else:
        file_regex = re.compile("^(Easy|Normal|Hard|Expert|ExpertPlus)\\.json$")
    json_files = []
    for root, subdirectories, files in os.walk(data_directory):
        for name in files:
            if bool(file_regex.match(name)):
                json_path = os.path.join(root, name)
                json_files.append(json_path)
    return json_files


def get_list_of_downloaded_songs():
    downloaded_songs_full = [dI for dI in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, dI))]
    i = 0
    downloaded_songs = []
    while i < len(downloaded_songs_full):
        try:
            downloaded_songs.append(downloaded_songs_full[i].split(')', 1))
        except:
            print('Fail @ ' + str(i) + ': ' + downloaded_songs_full[i])
            pass
        i += 1
    return downloaded_songs_full, downloaded_songs


def read_meta_data_file(filename):
    num_lines = sum(1 for line in open(filename))
    f = open(filename, 'r')
    meta_data = dict()
    meta_data['id'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['title'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['author'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['downloads'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['finished'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['thumbsUp'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['thumbsDown'] = f.readline().split(': ')[1].split('\n')[0]
    meta_data['rating'] = f.readline().split(': ')[1].split('\n')[0]
    if num_lines > 8:
        meta_data['scoresaberDifficulty'] = f.readline().split(': ')[1].split('\n')[0]
        if num_lines > 9:
            meta_data['scoresaberDifficultyLabel'] = f.readline().split(': ')[1].split('\n')[0]
            if num_lines > 10:
                meta_data['scoresaberId'] = f.readline().split(': ')[1].split('\n')[0]
                if num_lines > 11:
                    meta_data['funFactor'] = f.readline().split(': ')[1].split('\n')[0]
                    if num_lines > 12:
                        meta_data['rhythm'] = f.readline().split(': ')[1].split('\n')[0]
                        if num_lines > 13:
                            meta_data['flow'] = f.readline().split(': ')[1].split('\n')[0]
                            if num_lines > 14:
                                meta_data['patternQuality'] = f.readline().split(': ')[1].split('\n')[0]
                                if num_lines > 15:
                                    meta_data['readability'] = f.readline().split(': ')[1].split('\n')[0]
                                    if num_lines > 16:
                                        meta_data['levelQuality'] = f.readline().split(': ')[1].split('\n')[0]
    return meta_data


def write_meta_data_file(filename, meta_data):
    #if not os.path.exists(filename):
    #    f = open(filename, 'a').close()  # incase doesn't exist //

    # Write meta data text file
    f = open(filename, 'w')
    f.write('id: ' + html.unescape(meta_data['id']) + '\n')
    f.write('title: ' + html.unescape(meta_data['title']) + '\n')
    f.write('author: ' + html.unescape(meta_data['author']) + '\n')
    f.write('downloads: ' + html.unescape(meta_data['downloads']) + '\n')
    f.write('finished: ' + html.unescape(meta_data['finished']) + '\n')
    f.write('thumbsUp: ' + html.unescape(meta_data['thumbsUp']) + '\n')
    f.write('thumbsDown: ' + html.unescape(meta_data['thumbsDown']) + '\n')
    f.write('rating: ' + html.unescape(meta_data['rating']) + '\n')
    if 'scoresaberDifficulty' in meta_data.keys():
        del_list = []
        for i in range(len(meta_data['scoresaberDifficulty'])):
            if meta_data['scoresaberDifficulty'][i] is None:
                del_list.append(i)
        for i in range(len(del_list)-1, -1, -1):
            del meta_data['scoresaberId'][del_list[i]]
            del meta_data['scoresaberDifficulty'][del_list[i]]
            del meta_data['scoresaberDifficultyLabel'][del_list[i]]
        f.write('scoresaberDifficulty: ' + str(meta_data['scoresaberDifficulty']).replace('[', '').replace(']', '') + '\n')
    if 'scoresaberDifficultyLabel' in meta_data.keys():
        f.write('scoresaberDifficultyLabel: ' + str(meta_data['scoresaberDifficultyLabel']).replace('[', '').replace(']', '') + '\n')
    if 'scoresaberId' in meta_data.keys():
        f.write('scoresaberId: ' + str(meta_data['scoresaberId']).replace('[', '').replace(']', '') + '\n')
    if 'funFactor' in meta_data.keys():
        f.write('funFactor: ' + html.unescape(meta_data['funFactor']) + '\n')
    if 'rhythm' in meta_data.keys():
        f.write('rhythm: ' + html.unescape(meta_data['rhythm']) + '\n')
    if 'flow' in meta_data.keys():
        f.write('flow: ' + html.unescape(meta_data['flow']) + '\n')
    if 'patternQuality' in meta_data.keys():
        f.write('patternQuality: ' + html.unescape(meta_data['patternQuality']) + '\n')
    if 'readability' in meta_data.keys():
        f.write('readability: ' + html.unescape(meta_data['readability']) + '\n')
    if 'levelQuality' in meta_data.keys():
        f.write('levelQuality: ' + html.unescape(meta_data['levelQuality']) + '\n')
    f.close()
    return meta_data


def add_data_to_plot(x, y, title=None, ax=None, style='b-', label=None, legend=None, realtime=False):
    if(ax is None):
        fig, ax = pyplot.subplots()
    ax.plot(x, y, style, label=label)
    if(title is not None):
        ax.set_title(title)
    if(legend is not None):
        if legend is True:
            ax.legend()
        else:
            ax.legend(legend)
    if (realtime == True):
        pyplot.pause(0.05)
    return ax
