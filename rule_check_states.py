import numpy as np
import json
import math
from IOFunctions import *
from identifyStateSpace import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DATA_DIR = os.path.join(THIS_DIR, 'DataE')

def check_state_rules(data_directory, k):
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
            entry_row = entry[2]
            entry_column = entry[3]
            entry_time = entry[4]
            entry_type = entry[5]
            # The Computational Part
            entry_index = 4 * entry_column + entry_row  # Compute Index to update in the state representation
            if entry_type == 3:  # This is a bomb
                entry_representation = 19
            else:  # This is a note
                entry_representation = 1 + 9 * entry_type + entry_cut_direction
            # Now Retrieve and update the corresponding state representation
            state_dict[entry_time][entry_index] = entry_representation

        # for current song we will do some checks for good practices
        for i in note_times:
            # first check: vision blocks
            if((state_dict[i][5] != 0) and (state_dict[i][6] != 0)):
                print("Rule Violation: Vision Blocks")

            # second check: hammer hit
            if(19 in state_dict[i]):
                for x in range(0, 12):
                    if(state_dict[i][x] == 19):
                        bombIndex = x
                        break
                if(bombIndex == 0):
                    if(state_dict[i][4] in {2, 11} or state_dict[i][5] in {7, 16} or state_dict[i][1] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 1):
                    if(state_dict[i][0] in {4, 13} or state_dict[i][4] in {8, 17} or state_dict[i][5] in {2, 11}
                        or state_dict[i][6] in {7, 16} or state_dict[i][2] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 2):
                    if(state_dict[i][1] in {4, 13} or state_dict[i][5] in {8, 17} or state_dict[i][6] in {2, 11}
                        or state_dict[i][7] in {7, 16} or state_dict[i][3] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 3):
                    if(state_dict[i][2] in {4, 12} or state_dict[i][6] in {8, 17} or state_dict[i][7] in {2, 11}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 4):
                    if(state_dict[i][8] in {2, 11} or state_dict[i][9] in {7, 16} or state_dict[i][5] in {3, 12}
                        or state_dict[i][1] in {5, 14} or state_dict[i][0] in {1, 10}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 5):
                    if(state_dict[i][4] in {4, 13} or state_dict[i][8] in {8, 17} or state_dict[i][9] in {2, 11}
                        or state_dict[i][10] in {7, 16} or state_dict[i][6] in {3, 12} or state_dict[i][2] in {5, 14}
                        or state_dict[i][1] in {1, 10} or state_dict[i][0] in {6, 15}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 6):
                    if(state_dict[i][5] in {4, 13} or state_dict[i][9] in {8, 17} or state_dict[i][10] in {2, 11}
                        or state_dict[i][11] in {7, 16} or state_dict[i][7] in {3, 12} or state_dict[i][3] in {5, 14}
                        or state_dict[i][2] in {1, 10} or state_dict[i][1] in {6, 15}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 7):
                    if(state_dict[i][11] in {2, 11} or state_dict[i][10] in {8, 17} or state_dict[i][6] in {4, 13}
                        or state_dict[i][2] in {6, 15} or state_dict[i][3] in {1, 10}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 8):
                    if(state_dict[i][4] in {1, 10} or state_dict[i][5] in {5, 14} or state_dict[i][9] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 9):
                    if(state_dict[i][8] in {4, 13} or state_dict[i][4] in {6, 15} or state_dict[i][5] in {1, 10}
                        or state_dict[i][6] in {5, 14} or state_dict[i][10] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 10):
                    if(state_dict[i][9] in {4, 13} or state_dict[i][5] in {6, 15} or state_dict[i][6] in {1, 10}
                        or state_dict[i][7] in {5, 14} or state_dict[i][11] in {3, 12}):
                        print("Rule Violation: Hammer Hit")
                elif(bombIndex == 11):
                    if(state_dict[i][10] in {4, 13} or state_dict[i][6] in {6, 15} or state_dict[i][7] in {1, 10}):
                        print("Rule Violation: Hammer Hit")

            # third check: badly placed cut directions for adjacent notes
            # we are assuming at most there will be at most 4 notes, could also make this a check later
            # could also make a get adjacent notes function later
            numNotes = 0

            # find indices of adj notes
            for x in range(0, 12):
                if(state_dict[i][x] != 19 and state_dict[i][x] != 0):
                    # then we have a note and the indices of them!
                    numNotes += 1
                    if(numNotes == 1):
                        note_1 = x
                    elif(numNotes == 2):
                        note_2 = x
                        if (note_1 + 1 == note_2 and note_1 != 3 and note_1 != 7): #need to do some kind of mod check to see if they are on same row
                            # check their cutdirections
                            if((state_dict[i][note_1] ==  4 and state_dict[i][note_2] == 12) or (state_dict[i][note_1] == 13 and state_dict[i][note_2] == 3)):
                                print("Rule Violation: Controller Smash")
                            elif((state_dict[i][note_1] == 6 and state_dict[i][note_2] == 14) or (state_dict[i][note_1] == 15 and state_dict[i][note_2] == 5)):
                                print("Rule Violation: Controller Smash")
                            elif((state_dict[i][note_1] == 3 and state_dict[i][note_2] == 13) or (state_dict[i][note_1] == 12 and state_dict[i][note_2] == 4)):
                                print("Rule Violation: Impossible Pattern")
                    elif(numNotes == 3):
                        note_3 = x
                    elif(numNotes == 4):
                        note_4 = x
                        if(note_3 + 1 == note_4 and note_3 != 3 and note_3 != 7):
                            # check their cut directions
                            if ((state_dict[i][note_3] == 4 and state_dict[i][note_4] == 12) or
                                    (state_dict[i][note_3] == 13 and state_dict[i][note_4] == 3)):
                                print("Rule Violation: Controller Smash")
                            elif ((state_dict[i][note_3] == 6 and state_dict[i][note_4] == 14) or (state_dict[i][note_3] == 15 and state_dict[i][note_4] == 5)):
                                print("Rule Violation: Controller Smash")
                            elif ((state_dict[i][note_3] == 3 and state_dict[i][note_4] == 13) or (state_dict[i][note_3] == 12 and state_dict[i][note_4] == 4)):
                                print("Rule Violation: Impossible Pattern")

if __name__ == "__main__":
    check_state_rules(EXTRACTED_DATA_DIR, k=2000)












