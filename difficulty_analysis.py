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
x_1) Number of blocks
x_2) Number of unique blocks
x_3) Number of obstacles
x_4) Number of bombs
x_5) Beats per minute
x_6) Song length

y_0) Difficulty rating

z_0) Scoresaber difficulty rating

_Version 1_
Linear model: y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
'''

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