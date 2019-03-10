import IOFunctions, os, numpy as np
from collections import Counter

''' @RA: Trying to identify state patterns which appear in data, to determine an optimal state representation
Goal of this code:
1) Discretise the Beat Saber Levels into events at specific times
2) Identify "states" at each time and determine the distinct ones

Based on the findings of this experiment, we can determine which representation serves our objectives best :)

MAR 3 UPDATE: Code refactored to avoid state space computation code repetition
'''

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DATA_DIR = os.path.join(THIS_DIR, 'DataE')


def compute_explicit_states_from_json(level_json, as_tuple = True):
    json_representation = IOFunctions.parse_json(level_json)
    notes = json_representation["_notes"]  # Parse the JSON notes to use the notes representation
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
        return state_dict
    else: # Tuples can be hashed
        states_as_tuples = {time: tuple(state) for time, state in state_dict.items()}
        return states_as_tuples


def compute_shortest_inter_event_beat_gap(data_directory):
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(data_directory)
    minimum_beat_gap = np.inf
    for file in json_files:
        # print("Analysing file " + file)
        # Now go through them file by file
        json_representation = IOFunctions.parse_json(file)
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


def produce_distinct_state_space_representations(data_directory=EXTRACTED_DATA_DIR, k=2000):
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(data_directory)
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


def produce_transition_probability_matrix_from_distinct_state_spaces(states=None, data_directory=EXTRACTED_DATA_DIR):
    if states is None:
        states = produce_distinct_state_space_representations(2000, data_directory)
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(data_directory)
    for file in json_files:
        print("Analysing file " + file)
        transition_table = np.zeros((len(states), len(states)), dtype='uint8')
        state_dict = compute_explicit_states_from_json(file,as_tuple=False)
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


if __name__ == "__main__":
    sorted_states, states_counts = produce_distinct_state_space_representations(EXTRACTED_DATA_DIR, k=1000)
    sorted_states_prior_probability = np.divide(states_counts, sum(states_counts))
    output_path = os.path.join(THIS_DIR, 'stateSpace')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    IOFunctions.saveFile(sorted_states, 'sorted_states.pkl', output_path, append=False)
    IOFunctions.saveFile(sorted_states_prior_probability, 'sorted_states_prior_probability.pkl', output_path,
                         append=False)
    sorted_states_transition_probabilities = produce_transition_probability_matrix_from_distinct_state_spaces(
        sorted_states, EXTRACTED_DATA_DIR)
    IOFunctions.saveFile(sorted_states_transition_probabilities, 'sorted_states_transition_probabilities.pkl',
                         output_path, append=False)
    # compute_shortest_inter_event_beat_gap(EXTRACTED_DATA_DIR)
