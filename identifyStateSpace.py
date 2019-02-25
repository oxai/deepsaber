import IOFunctions, os, numpy as np
from collections import Counter
import matplotlib.pyplot as plt
''' @RA: Trying to identify state patterns which appear in data, to determine an optimal state representation 
Goal of this code: 
1) Discretise the Beat Saber Levels into events at specific times
2) Identify "states" at each time and determine the distinct ones 

Based on the findings of this experiment, we can determine which representation serves our objectives best :)
'''


def compute_inter_event_beat_gap_distribution(data_directory, max_gap = 1, bucketise = True):
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(data_directory)
    minimum_beat_gap = np.inf
    beat_gaps = []
    for file in json_files:
        #print("Analysing file " + file)
        # Now go through them file by file
        json_representation = IOFunctions.parse_json(file)
        notes = json_representation["_notes"]  #
        # Step 1: Extract the distinct times then convert to list and then numpy array for future processing.
        # Cumbersome, yes, but it works, and this is only for the sake of analytics
        note_times = np.sort(np.array(list(set(notes["_time"])))) # MUST BE SORTED
        event_onset_differences = np.diff(note_times)  # Step 2: Initialise the state representations
        event_onset_differences_filtered = event_onset_differences[event_onset_differences < 0.84] # Max_Gap
        event_onset_differences_filtered = event_onset_differences_filtered[event_onset_differences_filtered > 0.83]
        if bucketise: # Won't consider gaps smaller than 1/16th of a beat
            bucketised = np.ceil(np.log2(event_onset_differences_filtered)) # Reduce to numbers
            bucketised[bucketised < -6] = -6
            beat_gaps.extend(bucketised)
        # @RA Feb 12: Rather than compute the minimum beat gap, compute the distribution over beat gaps
        else:
            beat_gaps.extend(event_onset_differences_filtered)
        '''try:
            smallest_onset = np.min(event_onset_differences)
        except:
            smallest_onset = minimum_beat_gap + 1
            print("FLAG" + file)
        if smallest_onset < minimum_beat_gap:
            minimum_beat_gap = smallest_onset
    print(" The smallest beat gap between two events is " + str(minimum_beat_gap))'''
    gap_frequencies = Counter(beat_gaps)  # Count the frequency of every state
    x_axis = list(gap_frequencies.keys())
    y_axis = list(gap_frequencies.values()) # Well, there are some huge beat gaps
    # For 1/3rd analysis:
    print(len(beat_gaps))
    '''gap_frequencies_prominent = {x:y for x,y in gap_frequencies.items() if y > 500} # Identify the popular beat gaps
    print(gap_frequencies_prominent)'''
    plt.xlabel("Log 2 power of beat gap (discretised) ")
    plt.ylabel("Number of Occurrences")
    plt.yscale('log')
    plt.scatter(x=x_axis, y=y_axis, color='g')
    plt.show()
    # This is not indicative ... need to "bucket-ise"
    return gap_frequencies


def produce_distinct_state_space_representations(data_directory, k = 1000):
    json_files = IOFunctions.get_all_json_level_files_from_data_directory(data_directory)
    '''Definition of a state representation 
            @RA: This is a 12-dimensional array, such that every dimension represents a position in the grid
            If 0, position is empty, otherwise for a note: type * 9(numberOfDirections) + cutDirection + 1
            19: Bomb
        Statistics for states is very important
        
        Feb 12: Compute Top K states' total representation in overall state count
    '''
    list_of_states = [] # Initialise the set of states
    for file in json_files:
        print("Analysing file "+file)
        # Now go through them file by file
        json_representation = IOFunctions.parse_json(file)
        notes = json_representation["_notes"]  # Will Ignore Obstacles and Events for now
        # Step 1: Extract the DISTINCT times (hence the set structure)
        note_times = set(notes["_time"])  # Can Also Use This to compute minimal time difference between two note events
        # Step 2: Initialise the state representations
        state_dict = {eventTime: np.zeros(12) for eventTime in note_times}
        # Step 3: Now Go Through dataFrame entries and update states accordingly
        for entry in notes.itertuples():
            entry_cut_direction = entry[1]  # Extract the individual note parts
            entry_col = entry[2]
            entry_row = entry[3]
            entry_time = entry[4]
            entry_type = entry[5]
            # The Computational Part
            entry_index = 4 * entry_row + entry_col  # Compute Index to update in the state representation
            if entry_type == 3: # This is a bomb
                entry_representation = 19
            else: # This is a note
                entry_representation = 1 + 9 * entry_type + entry_cut_direction
            # Now Retrieve and update the corresponding state representation
            state_dict[entry_time][entry_index] = entry_representation
        # Now, all state representations at all events are computed and are stored (by time) in state_dict
        states = state_dict.values() # Get all state representations
        states_as_tuples = [tuple(i) for i in states]  # Convert to tuples (Needed to enable hashing)
        list_of_states.extend(states_as_tuples)  # Add to overall state list
    # Now all files are analysed, identify distinct sets
    total_nb_states = len(list_of_states)
    state_frequencies = Counter(list_of_states)  # Count the frequency of every state
    print(state_frequencies)
    distinct_states = state_frequencies.keys()  # Get distinct states. This avoids a set method which loses count info
    nb_of_distinct_states = len(distinct_states)
    distinct_state_frequencies = state_frequencies.values()
    '''count_distribution = Counter(distinct_state_frequencies)
    # print(count_distribution)'''
    # Sort Dictionary by number of states
    # We now have the states sorted by frequency
    sorted_states_by_frequency = sorted(state_frequencies, key=state_frequencies.get, reverse=True)
    sorted_values = sorted(distinct_state_frequencies,reverse=True)
    # Total number of states is len(list_of_states)
    top_k_representation = np.sum(sorted_values[:k])
    print(" We have "+str(nb_of_distinct_states)+" distinct states in our dataset")
    print(" Of these states, the top K="+str(k)+" of these represent "+str(100*top_k_representation/total_nb_states) +
          " % of the total state appearances")
    '''
    Next step : Compute Distribution over these states? Could be used as a reliability metric 
    How good is our generator? KL Divergence with the distribution?'''


if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    EXTRACTED_DATA_DIR = os.path.join(THIS_DIR, 'DataE')
    produce_distinct_state_space_representations(EXTRACTED_DATA_DIR, k=100)
    #compute_inter_event_beat_gap_distribution(EXTRACTED_DATA_DIR, bucketise=False)