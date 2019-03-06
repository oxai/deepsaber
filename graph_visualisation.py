import IOFunctions, os, numpy as np
from identifyStateSpace import compute_explicit_states_from_json
from graphviz import Digraph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DATA_DIR = os.path.join(THIS_DIR, 'DataE')

def produce_finite_state_machine_from_json(json_file):
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
    nodes = dict()
    transitions = dict()

    last_state = None
    for state in these_states:
        if last_state is not None:
            i = these_states_dict[tuple(state)]
            j = these_states_dict[tuple(last_state)]
            transition_table[i, j] += 1
        last_state = state
    transition_probabilities = []
    for i in range(len(these_states_dict.keys())):
        dot.node('q'+str(i), shape='circle')

    for i in range(len(transition_table)):
        transition_probabilities.append(np.divide(transition_table[i, :], sum(transition_table[i, :])))
        for j in range(len(transition_table[i])):
            if transition_table[i,j] > 0:
                dot.edge('q'+str(i), 'q'+str(j), label=str(transition_probabilities[i][j]))

    return dot


if __name__ == '__main__':
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(EXTRACTED_DATA_DIR)
    for json_file in json_files:
        graph_filename = json_file.split('.json')[0] + '_fsm'
        if not os.path.isfile(graph_filename+'.png'):
            dot = produce_finite_state_machine_from_json(json_file)
            view_output = False # True to view immediately
            dot.render(graph_filename, view=view_output, cleanup=True)

