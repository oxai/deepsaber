from decodeJSON import parse_json
from encodeJSON import create_dataStructure, encode_json
from featuresBase import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')


def generate_baseline_level_from_ogg(song_identifier, difficulty):
    difficulties = ['Easy', 'Normal', 'Hard']
    song_directory = os.path.join(EXTRACT_DIR, song_identifier)
    song_json = os.path.join(song_directory, difficulties[difficulty]+'.json')
    song_mod_json = os.path.join(song_directory, difficulties[difficulty]+'_mod.json')
    song_ogg = os.path.join(EXTRACT_DIR, song_directory, 'song.ogg')

    if not os.path.isfile(song_json):
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

    notes_mod = generate_beatsaber_notes_from_ogg(song_ogg, difficulty)
    events_mod = events
    obstacles_mod = obstacles

    #############################################
    # encode to json
    #############################################

    # create the right data format (dictionary within dictionary)
    data_to_json = create_dataStructure(events_mod, notes_mod, obstacles_mod, version, shufflePeriod, noteJumpSpeed,
                                        beatsPerBar, shuffle, bpm)
    # write to file
    json_test = encode_json(data_to_json, song_mod_json)


if __name__ == '__main__':
    for difficulty in range(3):
        generate_baseline_level_from_ogg('believer', difficulty)
