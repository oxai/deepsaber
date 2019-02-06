import json
import numpy as np
import pandas as pd


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
