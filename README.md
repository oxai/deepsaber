Google Doc: https://docs.google.com/document/d/1UDSphLiWsrbdr4jliFq8kzrJlUVKpF2asaL65GnnfoM/edit

# Example - see example_decodeEncode.py

from decodeJSON import parse_json

from encodeJSON import create_dataStructure, encode_json

from featuresBase import *

file_directory_song = 'C:\\Users\micha\Dropbox\\beatsaber\\believer\Believer\song.ogg'

file_directory = 'C:\\Users\micha\Dropbox\\beatsaber\\believer\Believer\Expert.json'

file_directory_encode = 'C:\\Users\micha\Dropbox\\beatsaber\\believer\Believer\Expert_mod.json'


# parse a json file and extrect the elements

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


# create new features


notes_mod = baseline_notes()

events_mod = events

obstacles_mod = obstacles


# encode to json

#create the right data format (dictionary within dictionary)

data_to_json = create_dataStructure(events_mod, notes_mod, obstacles_mod, version, shufflePeriod, noteJumpSpeed, beatsPerBar, shuffle, bpm)

#write to file

json_test = encode_json(data_to_json, file_directory_encode)
