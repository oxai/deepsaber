import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)

from scripts.misc import io_functions
import math
import numpy as np
import librosa
import os
import sys
import models.constants as constants
from collections import Counter

from scripts.feature_extraction.feature_extraction import extract_features_chroma, extract_features_mfcc, \
    extract_features_hybrid_beat_synced


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


def produce_distinct_state_space_representations(data_directory=EXTRACT_DIR, k=2000):
    '''Produces a list of all distinct states represented across all beatsaber levels found in the extracted data dir'''

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
    return sorted_states_by_frequency, sorted_states_count


def compute_explicit_states_from_json(level_json, as_tuple = True):
    '''Wrapper for extracting a BeatSaber Level state representation from JSON files.'''
    bs_level = io_functions.parse_json(level_json)
    states_as_tuples = compute_explicit_states_from_bs_level(bs_level, as_tuple)
    return states_as_tuples


def compute_explicit_states_from_bs_level(bs_level, as_tuple = True):
    '''Extract state representation from BeatSaber level.'''
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    note_times = set(notes["_time"])  # Extract the distinct note times
    state_dict = {eventTime: np.zeros(12) for eventTime in note_times}  # Initialise a state at every time event
    # for entry in notes.itertuples():
    for i,entry in notes.iterrows():
        entry_cut_direction = entry["_cutDirection"]  # Extract the individual note parts
        entry_col = entry["_lineIndex"]
        entry_row = entry["_lineLayer"]
        entry_time = entry["_time"]
        entry_type = entry["_type"]
        entry_index = int(4 * entry_row + entry_col)  # Compute Index to update in the state representation
        if entry_type == 3:  # This is a bomb
            entry_representation = 19
        else:  # This is a note
            entry_representation = 1 + 9 * entry_type + entry_cut_direction
        # Now Retrieve and update the corresponding state representation
        # print(entry_time, entry_index)
        try:
            state_dict[entry_time][entry_index] = entry_representation
        except:
            continue  # some weird notes with too big or small lineLayer / lineIndex ??
    if not as_tuple:
        return state_dict, note_times
    else:  # Tuples can be hashed
        states_as_tuples = {time: tuple(state) for time, state in state_dict.items()}
        return states_as_tuples


def compute_shortest_inter_event_beat_gap(data_directory):
    ''' Finds and returns the shortest time interval between beats of the beatsaber level.'''
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


if __name__ == "__main__":
    sorted_states, states_counts = produce_distinct_state_space_representations(EXTRACT_DIR, k=1000)
    sorted_states_prior_probability = np.divide(states_counts, sum(states_counts))
    output_path = DATA_DIR+"/statespace/"
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
