import json
import numpy as np
import pandas as pd
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(THIS_DIR, os.path.join('believer', os.path.join('Believer', 'Expert.json')))

def parse_json(file_directory):
    # input json file
    # returns: dataframes of: events, notes, obstacles, as well as additional parameters (in the form of a dictionary)
    json_data = open(file_directory).read()
    data = json.loads(json_data)
    events = data.get('_events')
    notes = data.get('_notes')
    obstacles = data.get('_obstacles')

    # create the dataframes
    df_events = pd.DataFrame(events)
    df_notes = pd.DataFrame(notes)
    df_obstacles = pd.DataFrame(obstacles)

    # update dataframes in the data file
    data['_events'] = df_events
    data['_notes'] = df_notes
    data['_obstacles'] = df_obstacles
    return data

if __name__ == '__main__':
    dict = parse_json(FILE_DIR)
    # main variables
    events = dict['_events']
    notes = dict['_notes']
    obstacles = dict['_obstacles']
    # additional information
    version = dict['_version']
    shufflePeriod = dict['_shufflePeriod']
    noteJumpSpeed = dict['_noteJumpSpeed']
    beatsPerBar = dict['_beatsPerBar']
    shuffle = dict['_shuffle']
    bpm = dict['_beatsPerMinute']
