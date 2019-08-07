from scripts.misc.io_functions import parse_json, create_dataStructure, encode_json
from scripts.feature_extraction.features_base import *
from shutil import copyfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.pardir(os.pardir(THIS_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
difficulties = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

def generate_baseline_level_from_ogg(song_identifier, difficulty):
    song_directory, song_ogg, song_json, song_filename = get_song_from_directory_by_identifier(song_identifier, difficulties[difficulty])
    output_directory = song_directory + '_nonML_mod'
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
        copyfile(song_ogg, os.path.join(output_directory, song_filename))
        try:
            copyfile(os.path.join(song_directory, 'cover.jpg'), os.path.join(output_directory, 'cover.jpg'))
        except Exception as e:
            print('Cover image might not exist')
            print(e)

    song_mod_json = os.path.join(output_directory, difficulties[difficulty] + '.json')

    if not os.path.isfile(song_json):
        print('Skipped file; No file for difficulty: '+difficulties[difficulty])
        return 0

    #############################################
    # parse a json file and extract the elements
    #############################################
    bsLevel = parse_json(song_json)

    # main variables

    events = bsLevel['_events']
    """ list of lists containing
        time: event time position in beats
        type: #event type ( [0,4] = Lighting Effects; [5,7] = Unused; 8 = Turning of large object in the middle;
                            9 = Zoom in effect; [10,11] = Unused; 12 = Move light 2; 13 = Move light 3; )
        value: #event value 
            [For event type 12 and 13] (0: stop moving; 1-infinity: speed for the movement)
            [For event type [0,4] (0: Off; [1,2]: Blue; 4: Unused?; 5-6: Red; 7: Red _ fade out"""

    notes = bsLevel['_notes']
    """ list of lists containing:
        time: Note time position in beats
        lineIndex: Note horizontal position ([0,3] start from left)
        lineLayer: Note vertical position ([0,2] start from bottom)
        type: Note type (0 = red, 1 = blue, 3 = bomb) 
        cutDirection: note cut direction (  0=up; 1=down; 2=left; 3=right; 
                                            4=up-left; 5=up-right; 6=down-left; 7=down-right; 
                                            8=no-direction"""
    obstacles = bsLevel['_obstacles']
    """ list of lists containing:
        time: Obstacle position in beats
        lineIndex: Obstacle horizontal position ([0,3] start from left)
        type: Obstacle type (0=wall; 1=ceiling)
        duration: Obstacle length in beats
        width: Obstacle width in lines (extend to the right)
    """

    # additional information
    version = bsLevel['_version']  # format version
    bpm = bsLevel['_beatsPerMinute']  # beats per minute
    beatsPerBar = bsLevel['_beatsPerBar']  # value only used by editor
    noteJumpSpeed = bsLevel['_noteJumpSpeed']  # movement speed of notes (now unused?)
    shuffle = bsLevel['_shuffle']  # how random notes colour are in no direction mode
    shufflePeriod = bsLevel['_shufflePeriod']  # how often notes should change color in no direction mode

    # unused information
    # time = bsLevel['_time'] # length of track
    # bookmarks = bsLevel['_bookmarks'] # ??? save points???

    ##############################################
    # create new features
    ##############################################

    pre_notes_mod = generate_beatsaber_notes_from_ogg(song_ogg, difficulty)
    events_mod = generate_beatsaber_events_from_ogg(song_ogg, difficulty)
       
    if (not difficulty == 1):
        obstacles_mod = generate_beatsaber_obstacles_from_ogg(song_ogg, difficulty)
    else:
        obstacles_mod = obstacles
        
    notes_mod = filter_generated_notes(pre_notes_mod,events_mod,obstacles_mod)

    #############################################
    # encode to json
    #############################################

    # create the right data format (dictionary within dictionary)
    data_to_json = create_dataStructure(events_mod, notes_mod, obstacles_mod, version, shufflePeriod, noteJumpSpeed,
                                        beatsPerBar, shuffle, bpm)
    # write to file
    json_test = encode_json(data_to_json, song_mod_json)


if __name__ == '__main__':
    for difficulty in range(1):
        generate_baseline_level_from_ogg('4)Believer\Believer', difficulty=3)
    print('done')