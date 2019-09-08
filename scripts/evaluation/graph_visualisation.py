import os, numpy as np

import sys

from scripts.misc import io_functions
from scripts.data_processing.state_space_functions import compute_explicit_states_from_json
from graphviz import Digraph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)

def produce_finite_state_machine_from_json(json_file, apply_filter=False):
    print("Analysing file " + json_file)
    state_dict = compute_explicit_states_from_json(json_file, as_tuple=False)
    these_states = state_dict.values()  # Get all state representations

    states_as_tuples = [tuple(i) for i in these_states]  # Convert to tuples (Needed to enable hashing)
    state_times = [i for i in state_dict.keys()]  # Get state_times
    #  Sort states and times by state times (might be unnecessary, but dictionaries are not sorted by default)
    sort_key = np.argsort(state_times)
    these_states = np.array(states_as_tuples)[sort_key]
    these_states_dict = {}
    for i in range(len(these_states)):
        if tuple(these_states[i]) not in these_states_dict.keys():
            these_states_dict.update({tuple(these_states[i]): len(these_states_dict.keys())})

    transition_table = np.zeros((len(these_states_dict), len(these_states_dict)), dtype='uint8')

    #these_states2 = sorted(states_as_tuples, key=list(state_times), reverse=True)

    dot = Digraph('FSM', format='png')
    nodes = []

    last_state = None
    for state in these_states:
        if last_state is not None:
            i = these_states_dict[tuple(state)]
            j = these_states_dict[tuple(last_state)]
            transition_table[i, j] += 1
        else:
            initial_nodes = [i]
        last_state = state
    transition_probabilities = []

    for i in range(len(transition_table)):
        transition_probabilities.append(np.divide(transition_table[i, :], sum(transition_table[i, :])))

    transition_probabilities = np.array(transition_probabilities) # for node deletion

    if apply_filter is True:
        for i in range(len(transition_probabilities)):
            if np.max(transition_probabilities[i]) > 0: # avoiding pure sink states with no outbound transitions
                transition_probabilities[i] = low_pass_filter_probabilities(transition_probabilities[i])
        identify_initial_states = True
        i = 0
        while identify_initial_states is True:
            if np.max(transition_probabilities[i, :]) < 1:
                identify_initial_states = False
            else:
                i = np.where(transition_probabilities[i, :] == 1)[0][0]
                initial_nodes.append(i)

        remove_nodes_with_no_predecessors = True
        i = 0
        delete_flag = False
        while remove_nodes_with_no_predecessors is True:
            if np.max(transition_probabilities[:, i]) == 0 and i not in initial_nodes:
                transition_probabilities = np.delete(transition_probabilities, i, 0)
                transition_probabilities = np.delete(transition_probabilities, i, 1)
                delete_flag = True
            i = i + 1
            if i >= len(transition_probabilities):
                if delete_flag is True:
                    delete_flag = False
                    i = 0
                else:
                    remove_nodes_with_no_predecessors = False

        #for i in range(len(transition_probabilities)):
            #transition_probabilities[i] = transition_probabilities[i] / np.sum(transition_probabilities[i])

    for i in range(len(transition_probabilities[0])):
        dot.node('q' + str(i), shape='circle')

    for i in range(len(transition_probabilities)):
        for j in range(len(transition_probabilities[i])):
            if transition_probabilities[i, j] > 0:
                dot.edge('q'+str(i), 'q'+str(j), label=str(transition_probabilities[i, j]))

    return dot

def low_pass_filter_probabilities(x):
    """Compute softmax values for each sets of scores in x."""
    x_valid = x[np.where(x > 0)]
    std = np.std(x_valid)
    max = np.max(x_valid)
    low_pass = (max - 2*std)
    x[np.where(x < low_pass)[0]] = 0
    return x


if __name__ == '__main__':
    json_files = io_functions.get_all_json_level_files_from_data_directory(EXTRACT_DIR)
    apply_filter = False
    view_output = False # True to view immediately
    for json_file in json_files:
        graph_filename = json_file.split('.json')[0] + '_fsm'
        if apply_filter is True:
            graph_filename += '_filtered'
        #if not os.path.isfile(graph_filename+'.png'):
        dot = produce_finite_state_machine_from_json(json_file, apply_filter)
        dot.render(graph_filename, view=view_output, cleanup=True)
