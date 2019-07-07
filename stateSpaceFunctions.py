import IOFunctions
from identifyStateSpace import compute_explicit_states_from_json
import math, numpy as np
import librosa
import os
#from featuresBase import extract_beat_times_chroma_tempo_from_ogg

'''
This file contains all helper functions to take a JSON level file and convert it to the current note representation
Requirements: The stateSpace directory. It contains sorted_states.pkl, which stores all identified states in the dataset.
To generate this folder, run identifyStateSpace.py
'''
NUM_DISTINCT_STATES = 4672 # This is the number of distinct states in our dataset
EMPTY_STATE_INDEX = 0 # or NUM_DISTINCT_STATES. CONVENTION: The empty state is the zero-th state.
SAMPLING_RATE = 16000
import base.Constants as Constants
NUM_SPECIAL_STATES=3 # also padding

def compute_state_sequence_representation_from_json(json_file, states=None, top_k=2000):
    '''
    :param json_file: The input JSON level file
    :param top_k: the top K states to keep (discard the rest)
    :return: The sequence of state ranks (of those in the top K) appearing in the level
    '''
    if states is None:  # Only load if state is not passed
        states = IOFunctions.loadFile("sorted_states.pkl", "stateSpace") # Load the state representation
    if EMPTY_STATE_INDEX == 0:  # RANK 0 is reserved for the empty state
        # states_rank = {state: i+1 for i, state in enumerate(states)}
        states_rank = {state: i+NUM_SPECIAL_STATES for i, state in enumerate(states)}
    else: # The empty state has rank NUM_DISTINCT_STATES
        states_rank = {state: i for i, state in enumerate(states)}
    explicit_states = compute_explicit_states_from_json(json_file)
    # Now map the states to their ranks (subject to rank being below top_k)
    state_sequence = {time: states_rank[exp_state] for time, exp_state in explicit_states.items()
                      if (exp_state in states_rank and states_rank[exp_state] <= top_k-1+NUM_SPECIAL_STATES)}
    return state_sequence


def get_block_sequence_with_deltas(json_file, song_length, bpm, step_size, top_k=2000,states=None,one_hot=False,return_state_times=False):
    '''
    :param json_file: The input JSON level file
    :param song_length: The length of input song (in seconds). Used to filter out states occurring after end of song, which can appear in levels
    :param bpm: The input song's beats per minute, to convert state times (which are in beats) to real-time
    :param top_k: the top K states to keep (discard the rest)
    :param step_size: The duration you aim to use to discretize state appearance time
    :param states: ???????
    :param one_hot: Returns states as one-hot (True) or numerical (False)
    :return_state_times: Returns the original beat occurrences of states, used for level variation within 2-stage model
    :return: 
    '''
    state_sequence = compute_state_sequence_representation_from_json(json_file=json_file, top_k=top_k, states=states)
    states_sequence_beat = [(time, state) for time, state in sorted(state_sequence.items(),key=lambda x:x[0]) if (time*60/bpm) <= song_length]
    states_sequence_real = [(time*60/bpm, state) for time, state in states_sequence_beat]
    times_real_extended = np.array([0] + [time for time, state in states_sequence_real] + [song_length])
    times_beats = np.array([time for time, state in states_sequence_beat])
    max_index = int(song_length/step_size)-1  # Ascertain that rounding at the last step doesn't create a state after end of song
    feature_indices = np.array([min(max_index,int((time/step_size)+0.5)) for time in times_real_extended])  # + 0.5 is for rounding
    # States in a level file are not necessarily in time order, so sorting is done here, while also 
    states = np.array([Constants.START_STATE]+[state for time, state in states_sequence_beat]+[Constants.END_STATE])
    if one_hot:
        adv_indexing_col = np.arange(len(states))  # Column used for advanced indexing to produce one-hot matrix
        one_hot_states = np.zeros((top_k + NUM_SPECIAL_STATES, states.shape[0]))
        one_hot_states[states.astype(int), adv_indexing_col.astype(int)] = 1  # Advanced Indexing to fill one hot
    time_diffs = np.diff(times_real_extended)  # Compute differences between times 
    delta_backward = np.expand_dims(np.insert(time_diffs, 0, times_real_extended[0]), axis=0)
    delta_forward = np.expand_dims(np.append(time_diffs, song_length - times_real_extended[-1]), axis=0)
    if one_hot:
        if return_state_times: # Return state beat times if requested
            return one_hot_states, states, times_beats, delta_forward, delta_backward, feature_indices
        else:
            return one_hot_states, states, delta_forward, delta_backward, feature_indices
    else:
        return states, delta_forward, delta_backward, feature_indices



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
    array_length = math.ceil(np.max(times)/beat_discretization) + 1 # DESIGN CHOICE: MAX TIME IS LAST STATE:

    # CAN MAKE THIS END OF SONG, BUT THIS WOULD INTRODUCE REDUNDANT 0 STATES
    output_sequence = np.full(array_length, EMPTY_STATE_INDEX)
    alternative_dict = {int(time/beat_discretization):state for time, state in state_sequence.items()}
    output_sequence[list(alternative_dict.keys())] = list(alternative_dict.values()) # Use advanced indexing to fill where needed
    '''
    Note About advanced indexing: Advanced indexing updates values based on order , so if index i appears more than once
    the latest appearance sets the value e.g. x[2,2] = 1,3 sets x[2] to 3, this means that , in the very unlikely event
    that two states are mapped to the same beat discretization, the later state survives.
    '''
    return output_sequence


def extract_all_representations_from_dataset(dataset_dir,top_k=2000,beat_discretization = 1/16):
    # Step 1: Identify All song directories
    song_directories = [os.path.join(dataset_dir,song_dir) for song_dir in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir,song_dir))]
    # Step 2: Pass each directory through the representation computation (and write code for saving obviously)
    directory_dict = {}
    for song_dir in song_directories:
        directory_dict[song_dir] = extract_representations_from_song_directory(song_dir,
                                        top_k=top_k,beat_discretization=beat_discretization, audio_feature_select=True)
        break
    return directory_dict    #TODO: Add some code here to save the representations eventually


def extract_representations_from_song_directory(directory,top_k=2000,beat_discretization=1/16, audio_feature_select="Hybrid"):
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
    y, sr = librosa.load(OGG_file)  # Load the OGG in LibROSA as usual
    level_state_feature_maps = {}
    for JSON_file in JSON_files: # Corresponding to different difficulty levels I hope
        bs_level = IOFunctions.parse_json(JSON_file)
        try:
            bpm = bs_level["_beatsPerMinute"] # Try to get BPM from metadata to avoid having to compute it from scratch
        except:
            y_harmonic, y_percussive = librosa.effects.hpss(y)  # Separate into two frequency channels
            bpm, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=None,  # Otherwise estimate
                                hop_length=512, start_bpm=120.0, tightness=100., trim=True, units='frames')
        # Compute State Representation
        level_states = compute_discretized_state_sequence_from_json(json_file=JSON_file,
                                                                    top_k=top_k,beat_discretization=beat_discretization)
        # NOTE: Some levels have states beyond the end of audio, so these must be trimmed, hence the following change
        length = int(librosa.get_duration(y) * (bpm / 60) / beat_discretization) + 1
        if length <= len(level_states):
            level_states = level_states[:length] # Trim to match the length of the song
        else: # Song plays on after final state
            level_states_2 = [EMPTY_STATE_INDEX]*length
            level_states_2[:len(level_states)] = level_states  # Add empty states to the end. New Assumption.
            level_states = level_states_2
        # print(len(level_states)) # Sanity Checks
        feature_extraction_times = [(i*beat_discretization)*(60/bpm) for i in range(len(level_states))]
        if audio_feature_select == "Chroma":  # for chroma
            audio_features = extract_features_chroma(y, sr, feature_extraction_times)
        elif audio_feature_select == "MFCC":  # for mfccs
            audio_features = extract_features_mfcc(y, sr, feature_extraction_times)
        elif audio_feature_select == "Hybrid":
            audio_features = extract_features_hybrid_beat_synced(y, sr, feature_extraction_times,bpm,beat_discretization)
        # chroma_features = chroma_feature_extraction(y,sr, feature_extraction_frames, bpm, beat_discretization)
        level_state_feature_maps[os.path.basename(JSON_file)] = (level_states, audio_features)

        # To obtain the chroma features for each pitch you access it like: chroma_features[0][0]
        # the first index number refers to the 12 pitches, so is indexes 0 to 11
        # the second index number refers to the chroma values, so is indexed from 0 to numOfStates - 1
        # print(audio_features.shape) # Sanity Check

        # WE SHOULD ALSO USE THE PERCUSSIVE FREQUENCIES IN OUR DATA, Otherwise the ML is losing valuable information
    return level_state_feature_maps


def stage_two_states_to_json_notes(state_sequence, state_times, bpm, hop, sr, state_rank=None):
    if state_rank is None:  # Only load if state is not passed
        # GUILLERMO: Provide the states rank yourself, otherwise let me know and we can change the
        # load script (i.e., feed ../stateSpace)
        state_rank = IOFunctions.loadFile("sorted_states.pkl", "stateSpace")   # Load the state representation
        # Add three all zero states for the sake of simplicity
    state_rank[0:0] = [tuple(12 * [0])] * 3  # Eliminate the need for conditionals
    states_grid = [state_rank[state] for state in state_sequence]
    if len(state_times) > len(states_grid):
        time_new = state_times[0:len(states_grid)]
    else:
        time_new = state_times
    notes = [grid_cell_to_json_note(grid_index, grid_value, time, bpm, hop, sr)
            for grid_state, time in zip(states_grid, time_new) for grid_index, grid_value in enumerate(grid_state)
            if grid_value > 0]

    return notes

def grid_cell_to_json_note(grid_index, grid_value, time, bpm, hop, sr):
    if grid_value > 0:  # Non-EMPTY grid cell
        # json_object = {"_time": (time * bpm * hop) / (sr * 60),
        # this is receiving bpm time actually :P
        json_object = {"_time": time,
                        "_lineIndex": int(grid_index % 4),
                       "_lineLayer": int(grid_index // 4)}
        if grid_value == 19:  # Bomb
            json_object["_type"] = 3
        else:  # Standard Block
            json_object["_type"] = int((grid_value - 1) // 9)
            json_object["_cutDirection"] = int((grid_value - 1) % 9)
        return json_object
    else:
        return None
