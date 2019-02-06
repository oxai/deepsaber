import json
import numpy as np
import pandas as pd

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

###########
# example
###########
'''
file_directory = 'C:\\Users\micha\Dropbox\\beatsaber\\believer\Believer\Expert.json'
dict = parse_json(file_directory)
#main variables
events = dict['_events']
notes = dict['_notes']
obstacles = dict['_obstacles']
#additional information
version = dict['_version']
shufflePeriod = dict['_shufflePeriod']
noteJumpSpeed = dict['_noteJumpSpeed']
beatsPerBar = dict['_beatsPerBar']
shuffle = dict['_shuffle']
bpm = dict['_beatsPerMinute']
'''