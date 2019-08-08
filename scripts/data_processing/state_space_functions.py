from scripts.misc import io_functions
from scripts.data_processing.state_space_functions import compute_explicit_states_from_json
import math
import numpy as np
import librosa
import os
import models.constants as constants
from collections import Counter

from scripts.feature_extraction.feature_extraction import extract_features_chroma, extract_features_mfcc, \
    extract_features_hybrid_beat_synced

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.pardir(os.pardir(THIS_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)


''' @RA: Trying to identify state patterns which appear in data, to determine an optimal state representation
Goal of this code:
1) Discretise the Beat Saber Levels into events at specific times
2) Identify "states" at each time and determine the distinct ones

Based on the findings of this experiment, we can determine which representation serves our objectives best :)

MAR 3 UPDATE: Code refactored to avoid state space computation code repetition

This file contains all helper functions to take a JSON level file and convert it to the current note representation
Requirements: The stateSpace directory. It contains sorted_states.pkl, which stores all identified states in the dataset.
'''

EMPTY_STATE_INDEX = 0  # or NUM_DISTINCT_STATES. CONVENTION: The empty state is the zero-th state.
NUM_SPECIAL_STATES = 3  # also padding


def compute_explicit_states_from_json(level_json, as_tuple = True):
    bs_level = io_functions.parse_json(level_json)
    states_as_tuples = compute_explicit_states_from_bs_level(bs_level, as_tuple)
    return states_as_tuples

def compute_explicit_states_from_bs_level(bs_level, as_tuple = True):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    note_times = set(notes["_time"])  # Extract the distinct note times
    state_dict = {eventTime: np.zeros(12) for eventTime in note_times}  # Initialise a state at every time event
    for entry in notes.itertuples():
        entry_cut_direction = entry[1]  # Extract the individual note parts
        entry_col = entry[2]
        entry_row = entry[3]
        entry_time = entry[4]
        entry_type = entry[5]
        entry_index = 4 * entry_row + entry_col  # Compute Index to update in the state representation
        if entry_type == 3:  # This is a bomb
            entry_representation = 19
        else:  # This is a note
            entry_representation = 1 + 9 * entry_type + entry_cut_direction
        # Now Retrieve and update the corresponding state representation
        state_dict[entry_time][entry_index] = entry_representation
    if not as_tuple:
        return state_dict, note_times
    else: # Tuples can be hashed
        states_as_tuples = {time: tuple(state) for time, state in state_dict.items()}
        return states_as_tuples


def compute_shortest_inter_event_beat_gap(data_directory):
    json_files = io_functions.get_all_json_level_files_from_data_directory(data_directory)
    minimum_beat_gap = np.inf
    for file in json_files:
        # print("Analysing file " + file)
        # Now go through them file by file
        json_representation = io_functions.parse_json(file)
        notes = json_representation["_notes"]  #
        # Step 1: Extract the distinct times then convert to list and then numpy array for future processing.
        # Cumbersome, yes, but it works, and this is only for the sake of analytics
        note_times = np.sort(np.array(list(set(notes["_time"]))))  # MUST BE SORTED
        event_onset_differences = np.diff(note_times)  # Step 2: Initialise the state representations
        try:
            smallest_onset = np.min(event_onset_differences)
        except:
            smallest_onset = minimum_beat_gap + 1
            print("FLAG" + file)

        if smallest_onset < minimum_beat_gap:
            minimum_beat_gap = smallest_onset
    print(" The smallest beat gap between two events is " + str(minimum_beat_gap))


def produce_distinct_state_space_representations(data_directory=EXTRACT_DIR, k=2000):
    json_files = io_functions.get_all_json_level_files_from_data_directory(data_directory)
    '''Definition of a state representation
            @RA: This is a 12-dimensional array, such that every dimension represents a position in the grid
            If 0, position is empty, otherwise for a note: type * 9(numberOfDirections) + cutDirection + 1
            19: Bomb
        Statistics for states is very important

        Feb 12: Compute Top K states' total representation in overall state count
    '''
    list_of_states = []  # Initialise the set of states
    for file in json_files:
        print("Analysing file " + file)
        state_dict = compute_explicit_states_from_json(file, as_tuple=True)
        states_as_tuples = state_dict.values()
        list_of_states.extend(states_as_tuples)  # Add to overall state list

    # Now all files are analysed, identify distinct sets
    total_nb_states = len(list_of_states)
    state_frequencies = Counter(list_of_states)  # Count the frequency of every state
    distinct_states = state_frequencies.keys()  # Get distinct states. This avoids a set method which loses count info
    nb_of_distinct_states = len(distinct_states)
    distinct_state_frequencies = state_frequencies.values()
    # Sort Dictionary by number of states
    # We now have the states sorted by frequency
    sorted_states_by_frequency = sorted(state_frequencies, key=state_frequencies.get, reverse=True)
    sorted_states_count = sorted(distinct_state_frequencies, reverse=True)
    # Total number of states is len(list_of_states)
    top_k_representation = np.sum(sorted_states_count[:k])
    print(" We have " + str(nb_of_distinct_states) + " distinct states in our dataset")
    print(" Of these states, the top K=" + str(k) + " of these represent " + str(
        100 * top_k_representation / total_nb_states) +
          " % of the total state appearances")
    '''
    Next step : Compute Distribution over these states? Could be used as a reliability metric
    How good is our generator? KL Divergence with the distribution?'''
    return sorted_states_by_frequency, sorted_states_count


def produce_transition_probability_matrix_from_distinct_state_spaces(states=None, data_directory=EXTRACT_DIR):
    if states is None:
        states = produce_distinct_state_space_representations(2000, data_directory)
    json_files = io_functions.get_all_json_level_files_from_data_directory(data_directory)
    for file in json_files:
        print("Analysing file " + file)
        transition_table = np.zeros((len(states), len(states)), dtype='uint8')
        state_dict, __ = compute_explicit_states_from_json(file,as_tuple=False)
        these_states = state_dict.values()  # Get all state representations
        states_as_tuples = [tuple(i) for i in these_states]  # Convert to tuples (Needed to enable hashing)
        state_times = [i for i in state_dict.keys()]  # Get state_times
        #  Sort states and times by state times (might be unnecessary, but dictionaries are not sorted by default)
        sort_key = np.argsort(state_times)
        these_states = np.array(states_as_tuples)[sort_key]
        # these_states = sorted(states_as_tuples, key=sort_key, reverse=True)

        last_state = None
        for state in these_states:
            if last_state is not None:
                j = states.index(tuple(state))
                i = states.index(tuple(last_state))
                transition_table[i, j] += 1
            last_state = state

    transition_probabilities = []
    for i in range(len(transition_table)):
        transition_probabilities.append(np.divide(transition_table[i, :], sum(transition_table[i, :])))

    return transition_probabilities

def compute_state_sequence_representation_from_json(json_file, states=None, top_k=2000):
    '''
    :param json_file: The input JSON level file
    :param top_k: the top K states to keep (discard the rest)
    :return: The sequence of state ranks (of those in the top K) appearing in the level
    '''
    if states is None:  # Only load if state is not passed
        states = io_functions.loadFile("sorted_states.pkl", "stateSpace") # Load the state representation
    if EMPTY_STATE_INDEX == 0:  # RANK 0 is reserved for the empty state
        # states_rank = {state: i+1 for i, state in enumerate(states)}
        states_rank = {state: i+NUM_SPECIAL_STATES for i, state in enumerate(states)}
    else:
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
    states = np.array([constants.START_STATE]+[state for time, state in states_sequence_beat]+[constants.END_STATE])
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
                                        top_k=top_k, beat_discretization=beat_discretization, audio_feature_select=True)
        break
    return directory_dict    #TODO: Add some code here to save the representations eventually


def extract_representations_from_song_directory(directory, top_k=2000, beat_discretization=1/16, audio_feature_select="Hybrid"):
    OGG_files = io_functions.get_all_ogg_files_from_data_directory(directory)
    if len(OGG_files) == 0:  # No OGG file ... skip
        print("No OGG file for song "+directory)
        return
    OGG_file = OGG_files[0]  # There should only be one OGG file in every directory anyway, so we get that
    JSON_files = io_functions.get_all_json_level_files_from_data_directory(directory)
    if len(JSON_files) == 0:  # No Non-Autosave JSON files
        JSON_files = io_functions.get_all_json_level_files_from_data_directory(directory, include_autosaves=True)
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
        bs_level = io_functions.parse_json(JSON_file)
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
        state_rank = io_functions.loadFile("sorted_states.pkl", "stateSpace")   # Load the state representation
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


if __name__ == "__main__":
    sorted_states, states_counts = produce_distinct_state_space_representations(EXTRACT_DIR, k=1000)
    sorted_states_prior_probability = np.divide(states_counts, sum(states_counts))
    output_path = os.path.join(THIS_DIR, 'stateSpace')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    io_functions.saveFile(sorted_states, 'sorted_states.pkl', output_path, append=False)
    io_functions.saveFile(sorted_states_prior_probability, 'sorted_states_prior_probability.pkl', output_path,
                          append=False)
    sorted_states_transition_probabilities = produce_transition_probability_matrix_from_distinct_state_spaces(
        sorted_states, EXTRACT_DIR)
    io_functions.saveFile(sorted_states_transition_probabilities, 'sorted_states_transition_probabilities.pkl',
                          output_path, append=False)
    # compute_shortest_inter_event_beat_gap(EXTRACT_DIR)
