import IOFunctions
from identifyStateSpace import compute_explicit_states_from_json
import math, numpy as np
import librosa
import os
from featuresBase import extract_beat_times_chroma_tempo_from_ogg
'''
This file contains all helper functions to take a JSON level file and convert it to the current note representation
Requirements: The stateSpace directory. It contains sorted_states.pkl, which stores all identified states in the dataset.
To generate this folder, run identifyStateSpace.py
'''
NUM_DISTINCT_STATES = 4672 # This is the number of distinct states in our dataset
EMPTY_STATE_INDEX = 0 # or NUM_DISTINCT_STATES. CONVENTION: The empty state is the zero-th state.


def compute_state_sequence_representation_from_json(json_file, top_k=2000):
    '''

    :param json_file: The input JSON level file
    :param top_k: the top K states to keep (discard the rest)
    :return: The sequence of state ranks (of those in the top K) appearing in the level
    '''
    states = IOFunctions.loadFile("sorted_states.pkl", "stateSpace") # Load the state representation
    if EMPTY_STATE_INDEX == 0: # RANK 0 is reserved for the empty state
        states_rank = {state: i+1 for i, state in enumerate(states)}
    else: # The empty state has rank NUM_DISTINCT_STATES
        states_rank = {state: i for i, state in enumerate(states)}
    explicit_states = compute_explicit_states_from_json(json_file)
    # Now map the states to their ranks (subject to rank being below top_k)
    state_sequence = {time: states_rank[exp_state] for time, exp_state in explicit_states.items()
                      if states_rank[exp_state] <= top_k}
    return state_sequence


def compute_discretized_state_sequence_from_json(json_file, top_k=2000,beat_discretization = 1/16):
    '''
    :param json_file: The input JSON level file
    :param top_k: the top K states to keep (discard the rest)
    :param beat_discretization: The beat division with which to discretize the sequence
    :return: a sequence of state ranks discretised to the beat division
    '''
    state_sequence = compute_state_sequence_representation_from_json(json_file=json_file, top_k=top_k)
    # Compute length of sequence array. Clearly as discretization drops, length increases
    times = list(state_sequence.keys())
    array_length = math.ceil(np.max(times)/beat_discretization)  # DESIGN CHOICE: MAX TIME IS LAST STATE:
    # CAN MAKE THIS END OF SONG, BUT THIS WOULD INTRODUCE REDUNDANT 0 STATES
    output_sequence = np.full(array_length, EMPTY_STATE_INDEX)
    alternative_dict = {int(time/beat_discretization):state for time, state in state_sequence.items()}
    output_sequence[list(alternative_dict.keys())] = list(alternative_dict.values()) # Use advanced indexing to fill where needed
    '''
    Note About advanced indexing: Advanced indexing updates values based on order , so if index i appears more than once
    the latest appearance sets the value e.g. x[2,2] = 1,3 sets x[2] to 3, this means that , in the very unlikely event 
    that two states are mapped to the same beat discretization, the later state survives.
    '''
    print(list(output_sequence))
    return output_sequence


def extract_all_representations_from_dataset(dataset_dir,top_k=2000,beat_discretization = 1/16):
    # Step 1: Identify All song directories
    song_directories = [os.path.join(dataset_dir,song_dir) for song_dir in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir,song_dir))]
    # Step 2: Pass each directory through the representation computation (and write code for saving obviously)
    for song_dir in song_directories:
        extract_representations_from_song_directory(song_dir,top_k=top_k,beat_discretization=beat_discretization)
        break
        #TODO: Add some code here to save the representations eventually


def extract_representations_from_song_directory(directory,top_k=2000,beat_discretization=1/16):
    OGG_files = IOFunctions.get_all_ogg_files_from_data_directory(directory)
    if len(OGG_files) == 0:  # No OGG file ... skip
        print("No OGG file for song "+directory)
        return
    OGG_file = OGG_files[0]  # There should only be one OGG file in every directory anyway, so we get that
    JSON_files = IOFunctions.get_all_json_level_files_from_data_directory(directory)
    if len(JSON_files) == 0:  # No Non-Autosave JSON files
        JSON_files = IOFunctions.get_all_json_level_files_from_data_directory(directory, include_autosaves=True)
        # So now it's worth checking out the autosaves
        if len(JSON_files) == 0: # If there's STILL no JSON file, declare failure (some levels only have autosave)
            print("No level data for song "+directory)
            return
        else:
            JSON_files = [JSON_files[0]] # Only get the first element in case of autosave-only
            # (they're usually the same level saved multiple times so no point)

    # We now have all the JSON and OGGs for a level (if they exist). Process them
    # Feature Extraction Begins
    y, fs = librosa.load(OGG_file, sr=None)  # Load the OGG in LibROSA as usual
    y_harmonic, y_percussive = librosa.effects.hpss(y)  # Separate into two frequency channels
    for JSON_file in JSON_files: # Corresponding to different difficulty levels I hope
        bs_level = IOFunctions.parse_json(JSON_file)
        try:
            bpm = bs_level["_beatsPerMinute"] # Try to get BPM from metadata to avoid having to compute it from scratch
        except:
            bpm, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs, onset_envelope=None,  # Otherwise estimate
                                hop_length=512, start_bpm=120.0, tightness=100., trim=True, units='frames')
        # Compute State Representation
        level_states = compute_discretized_state_sequence_from_json(top_k=top_k,beat_discretization=beat_discretization)
        feature_extraction_times = [(i*beat_discretization)*(60/bpm) for i in range(len(level_states))]
        chroma_features = chroma_feature_extraction(y, feature_extraction_times, bpm, beat_discretization)
        # WE SHOULD ALSO USE THE PERCUSSIVE FREQUENCIES IN OUR DATA, Otherwise the ML is losing valuable information
        print(level_states)
        print(chroma_features)
        return level_states, chroma_features


def chroma_feature_extraction(y, state_times, bpm, beat_discretization = 1/16):
    hop = int((44100 * 60 * beat_discretization) / bpm)
    chromagram = librosa.feature.chroma_cqt(y=y, sr=fs, C=None, hop_length=hop, fmin=None,
                                            norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
                                            n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full')

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram, state_times, aggregate=np.median, pad=True, axis=-1)
    beat_chroma = beat_chroma[:, :-1]

    return beat_chroma





