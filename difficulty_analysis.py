from urllib.request import Request, urlopen
import os
import re
import html

from IOFunctions import read_meta_data_file, get_list_of_downloaded_songs

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

''' @TS: Trying to regress a model of difficulty from beatsaber level features:
x_00) Number of blocks
x_01) Number of unique blocks
x_02) Number of obstacles
x_03) Number of bombs
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

def extract_features_from_beatsaber_level(bsLevel):
    #feature_1 = distance travelled
    #feature_2 = angles travelled
    #feature_3 = sum of product distance/angle vectors

    num_blocks = extract_level_num_blocks(bsLevel)
    num_unique_blocks = extract_level_num_unique_blocks(bsLevel)
    num_obstacles = extract_level_num_obstacles(bsLevel)
    num_bombs = extract_level_num_bombs(bsLevel)

    beats_per_minute = extract_level_beats_per_minute(bsLevel)
    blocks_per_minute = extract_level_blocks_per_minute(bsLevel)
    blocks_per_beat = extract_level_blocks_per_beat(bsLevel)
    song_length = extract_level_song_length(bsLevel)

    distance_travelled = extract_level_distance_travelled(bsLevel)
    angles_travelled = extract_level_angles_travelled(bsLevel)
    product_distance_travelled = extract_level_product_distance_travelled(bsLevel)

    return num_blocks, num_unique_blocks, num_obstacles, num_bombs,\
           beats_per_minute, blocks_per_minute, blocks_per_beat, song_length, \
           distance_travelled, angles_travelled, product_distance_travelled

def extract_level_num_blocks(bsLevel):
    return 0

def extract_level_num_unique_blocks(bsLevel):
    return 0

def extract_level_num_obstacles(bsLevel):
    return 0

def extract_level_num_bombs(bsLevel):
    return 0

def extract_level_beats_per_minute(bsLevel):
    return 0

def extract_level_blocks_per_minute(bsLevel):
    return 0

def extract_level_blocks_per_beat(bsLevel):
    return 0

def extract_level_song_length(bsLevel):
    return 0

def extract_level_distance_travelled(bsLevel):
    return 0

def extract_level_angles_travelled(bsLevel):
    return 0

def extract_level_product_distance_travelled(bsLevel):
    return 0

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