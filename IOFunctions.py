import json, re
import pandas as pd
import os
import time
import getpass
import pickle
from glob import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

difficulties = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

def loadFile(filename):
    print('Loading file: ' + filename)
    loadfile = open(filename, 'rb')
    object = pickle.load(loadfile)
    loadfile.close()
    return object


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

def parse_json(file_directory):
    # input json file
    # returns: dataframes of: events, notes, obstacles, as well as additional parameters (in the form of a dictionary)
    json_data = open(file_directory).read()
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
    if len(os.listdir(song_directory)) == 1:
        song_directory = os.path.join(song_directory, os.listdir(song_directory)[0])
    song_json = dict()
    if difficulty is None:
        for difficulty in difficulties:
            json = os.path.join(song_directory, difficulty + '.json')
            if os.path.isfile(json):
                song_json[difficulty] = json
    else:
        json = os.path.join(song_directory, difficulty + '.json')
        if os.path.isfile(json):
            song_json = json
        else:
            raise Exception('No file of difficulty: '+difficulty+' in: '+song_directory)
    song_ogg = glob(os.path.join(song_directory, '*.ogg'))[0]
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