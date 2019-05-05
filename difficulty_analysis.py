from urllib.request import Request, urlopen
import os
import re
import html

import IOFunctions
from IOFunctions import read_meta_data_file, get_list_of_downloaded_songs, get_all_json_level_files_from_data_directory
from identifyStateSpace import compute_explicit_states_from_bs_level

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

''' @TS: Trying to regress a model of difficulty from beatsaber level features:
x_00) Number of blocks
x_01) Number of obstacles
x_02) Number of bombs
x_04) Number of unique states (bombs + blocks)
x_04) Beats per minute
x_05) Blocks per minute
x_06) Blocks per beat
x_07) Song length
x_08) Distance Travelled
x_09) Angles Travelled
x_10) Product Distance Travelled

y_0) Difficulty rating

z_0) Scoresaber difficulty rating

_Version 1_
Linear model: y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
'''

def extract_features_from_beatsaber_level(bs_level):
    #feature_1 = distance travelled
    #feature_2 = angles travelled
    #feature_3 = sum of product distance/angle vectors

    num_blocks = extract_level_num_blocks(bs_level)
    num_obstacles = extract_level_num_obstacles(bs_level)
    num_bombs = extract_level_num_bombs(bs_level)
    num_unique_states = extract_level_num_unique_states(bs_level)

    beats_per_minute = extract_level_beats_per_minute(bs_level)
    blocks_per_minute = extract_level_blocks_per_minute(bs_level)
    blocks_per_beat = extract_level_blocks_per_beat(bs_level)
    song_length = extract_level_song_length(bs_level)

    [distance_acc_blue, distance_acc_red, velocity_blue, velocity_red] = extract_level_distance_velocity(bs_level)
    angles_travelled = extract_level_angles_travelled(bs_level)
    product_distance_travelled = extract_level_product_distance_travelled(bs_level)

    return [num_blocks, num_obstacles, num_bombs, num_unique_states,\
           beats_per_minute, blocks_per_minute, blocks_per_beat, song_length, \
           distance_travelled, angles_travelled, product_distance_travelled]


def extract_level_num_blocks(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    num_blocks = 0
    for entry in notes.itertuples():
        if entry[5] != 3:  # This is not a bomb
            num_blocks += 1
    return num_blocks


def extract_level_num_obstacles(bs_level):
    return bs_level['_obstacles']['values'].shape[0]


def extract_level_num_bombs(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    num_bombs = 0
    for entry in notes.itertuples():
        if entry[5] == 3:  # This is a bomb
            num_bombs += 1
    return num_bombs


def extract_level_num_unique_states(bs_level):
    return len(compute_explicit_states_from_bs_level(bs_level))


def extract_level_beats_per_minute(bs_level):
    return bs_level['_beatsPerMinute']


def extract_level_blocks_per_minute(bs_level):
    return extract_level_num_blocks(bs_level) / extract_level_song_length(bs_level)


def extract_level_blocks_per_beat(bs_level):
    return extract_level_blocks_per_minute(bs_level) / extract_level_beats_per_minute(bs_level)


def extract_level_song_length(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    note_times = set(notes["_time"])
    return max(note_times)


def extract_level_distance_velocity(bs_level):
    #input is a json file with beatsaber data, json path
    #event Type is either: "events", "notes", "obstacles"

    df_notes = bs_level['_notes']
    bpm = bs_level['_beatsPerMinute']

    # separate the red an blue blocks
    array_blue = df_notes.loc[array['_type'] == 0]
    array_red = df_notes.loc[array['_type']==1]
    # event value = 0 or 1 (red and blue)
    [distance_acc_blue, velocity_blue] = return_distance_velocity(array_blue)
    [distance_acc_red, velocity_red] = return_distance_velocity(array_red)
    return distance_acc_blue, distance_acc_red, velocity_blue, velocity_red


def return_distance_velocity(df):
    #initialize starting positions
    tMinus1 = 0
    rowMinus1 =0
    columnMinus1 =0
    distance_acc = 0
    df = df.sort_values('_time')
    counter = 0
    for index, element in df.iterrows():
        counter += 1
        t = element['_time']
        row = element['_lineIndex']
        column = element['_lineLayer']
        distance = ((row-rowMinus1)**2+(column-columnMinus1)**2)**(1/2)
        dt = t-tMinus1
        if dt ==0: #because there are some blocks postioned right next to each other at the same time.
            dt = 0.1
        #velocity = distance/(dt)*bpm #convert to blocks per beat to blocks per second
        #velocity_avg = distance_acc/t*bpm
        rowMinus1 = row
        columnMinus1 = column
        tMinus1 = t
        distance_acc += distance

    velocity_avg = distance_acc/t

    return distance_acc, velocity_avg


def extract_level_angles_travelled(bs_level):
    return 0


def extract_level_product_distance_travelled(bs_level):
    return 0


#actually not really need...oh well
def convert_lin_col_to_coordinates(lineIndex, lineLayer):
    # takes as input the line and the column of the element, and returns an array 3x4, with 1 for the entry, where the element
    noteArray = np.zeros((3,4))
    for j in range(0,4):
        for i in range(0,3):
            if i == lineIndex and j == lineLayer:
                noteArray[i, j] = 1
    return noteArray

#def generate_difficulty_features():
    # Generating Data Frame
    # Step 1 - Load json file
    # Read:     Number of blocks
    #           Number of unique blocks
    #           Number of obstacles
    #           Number of bombs

    # Step 2 - Load ogg file
    # Read:     Beats per minute
    #           Song length

    # Step 3 - Load difficulty rating
    # Read:     Difficulty Rating
    #           Number of votes


if __name__ == '__main__':
    downloaded_songs_full, downloaded_songs = get_list_of_downloaded_songs()
    for song_dir in downloaded_songs_full:
        meta_data_filename = os.path.join(EXTRACT_DIR, os.path.join(song_dir, 'meta_data.txt'))
        if not os.path.exists(meta_data_filename):
            pass
        else:
            meta_data = read_meta_data_file(meta_data_filename)
            json_files = get_all_json_level_files_from_data_directory(os.path.join(EXTRACT_DIR, song_dir))
            for json_file in json_files:
                bs_level = IOFunctions.parse_json(json_file)
                features = extract_features_from_beatsaber_level(bs_level)
