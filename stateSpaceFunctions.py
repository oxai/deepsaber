import IOFunctions
from identifyStateSpace import compute_explicit_states_from_json
import math, numpy as np
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
    array_length = math.ceil(np.max(times)/beat_discretization)
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

