import numpy as np
import json
import math
from io_functions import *
from identify_state_space import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DATA_DIR = os.path.join(THIS_DIR, 'DataE')

def check_state_rules_for_directory(data_directory):
    json_files = io_functions.get_all_json_level_files_from_data_directory(data_directory)
    '''Definition of a state representation 
            @RA: This is a 12-dimensional array, such that every dimension represents a position in the grid
            If 0, position is empty, otherwise for a note: type * 9(numberOfDirections) + cutDirection + 1
            19: Bomb
        Statistics for states is very important
    '''
    level_validations = []

    for file in json_files:
        level_validation, _, _ = check_state_rules_of_bsLevel(file)
        level_validations.append(level_validation)

    return zip(json_files, level_validations)


def check_state_rules_of_bsLevel(file):
    print("Analysing file " + file)

    # use function from identify_state_space to extract states from json
    state_dict, note_times = compute_explicit_states_from_json(file, as_tuple=False)

    state_validations = []
    validation_codes = []
    level_validation = True

    # for current song we will do some checks for good practices
    for state, note_time in zip(state_dict.values(), list(note_times)):
        state_validation, validation_code = verify_state_rule_check(state)
        state_validations.append(state_validation)
        validation_codes.append(validation_code)
        if state_validation == False:
            level_validation = state_validation

    return level_validation, state_validations, validation_codes

def verify_state_rule_check(state):

    state_validation_code = 0

    state_validation_messages = {
        0: "State Validation Successful",
        1: "Rule Violation: Vision Blocks",
        2: "Rule Violation: Hammer Hit",
        3: "Rule Violation: Controller Smash",
        4: "Rule Violation: Impossible Pattern"
    }

    # first check: vision blocks
    if ((state[5] != 0) and (state[6] != 0)):
        state_validation_code = 1

    # second check: hammer hit
    if (19 in state):
        for x in range(0, 12):
            if (state[x] == 19):
                bombIndex = x
                break
        if (bombIndex == 0):
            if (state[4] in {2, 11} or state[5] in {7, 16} or state[1] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 1):
            if (state[0] in {4, 13} or state[4] in {8, 17} or state[5] in {2, 11}
                    or state[6] in {7, 16} or state[2] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 2):
            if (state[1] in {4, 13} or state[5] in {8, 17} or state[6] in {2, 11}
                    or state[7] in {7, 16} or state[3] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 3):
            if (state[2] in {4, 12} or state[6] in {8, 17} or state[7] in {2, 11}):
                state_validation_code = 2
        elif (bombIndex == 4):
            if (state[8] in {2, 11} or state[9] in {7, 16} or state[5] in {3, 12}
                    or state[1] in {5, 14} or state[0] in {1, 10}):
                state_validation_code = 2
        elif (bombIndex == 5):
            if (state[4] in {4, 13} or state[8] in {8, 17} or state[9] in {2, 11}
                    or state[10] in {7, 16} or state[6] in {3, 12} or state[2] in {5, 14}
                    or state[1] in {1, 10} or state[0] in {6, 15}):
                state_validation_code = 2
        elif (bombIndex == 6):
            if (state[5] in {4, 13} or state[9] in {8, 17} or state[10] in {2, 11}
                    or state[11] in {7, 16} or state[7] in {3, 12} or state[3] in {5, 14}
                    or state[2] in {1, 10} or state[1] in {6, 15}):
                state_validation_code = 2
        elif (bombIndex == 7):
            if (state[11] in {2, 11} or state[10] in {8, 17} or state[6] in {4, 13}
                    or state[2] in {6, 15} or state[3] in {1, 10}):
                state_validation_code = 2
        elif (bombIndex == 8):
            if (state[4] in {1, 10} or state[5] in {5, 14} or state[9] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 9):
            if (state[8] in {4, 13} or state[4] in {6, 15} or state[5] in {1, 10}
                    or state[6] in {5, 14} or state[10] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 10):
            if (state[9] in {4, 13} or state[5] in {6, 15} or state[6] in {1, 10}
                    or state[7] in {5, 14} or state[11] in {3, 12}):
                state_validation_code = 2
        elif (bombIndex == 11):
            if (state[10] in {4, 13} or state[6] in {6, 15} or state[7] in {1, 10}):
                state_validation_code = 2

    # third check: badly placed cut directions for adjacent notes
    # we are assuming at most there will be at most 4 notes, could also make this a check later
    # could also make a get adjacent notes function later
    numNotes = 0

    # find indices of adj notes
    for x in range(0, 12):
        if (state[x] != 19 and state[x] != 0):
            # then we have a note and the indices of them!
            numNotes += 1
            if (numNotes == 1):
                note_1 = x
            elif (numNotes == 2):
                note_2 = x
                if (note_1 + 1 == note_2 and note_1 != 3 and note_1 != 7):  # need to do some kind of mod check to see if they are on same row
                    # check their cutdirections
                    if ((state[note_1] == 4 and state[note_2] == 12) or (
                            state[note_1] == 13 and state[note_2] == 3)):
                        state_validation_code = 3
                    elif ((state[note_1] == 6 and state[note_2] == 14) or (
                            state[note_1] == 15 and state[note_2] == 5)):
                        state_validation_code = 3
                    elif ((state[note_1] == 3 and state[note_2] == 13) or (
                            state[note_1] == 12 and state[note_2] == 4)):
                        state_validation_code = 4
            elif (numNotes == 3):
                note_3 = x
            elif (numNotes == 4):
                note_4 = x
                if (note_3 + 1 == note_4 and note_3 != 3 and note_3 != 7):
                    # check their cut directions
                    if ((state[note_3] == 4 and state[note_4] == 12) or
                            (state[note_3] == 13 and state[note_4] == 3)):
                        state_validation_code = 3
                    elif ((state[note_3] == 6 and state[note_4] == 14) or (
                            state[note_3] == 15 and state[note_4] == 5)):
                        state_validation_code = 3
                    elif ((state[note_3] == 3 and state[note_4] == 13) or (
                            state[note_3] == 12 and state[note_4] == 4)):
                        state_validation_code = 4

    if state_validation_code != 0:
        print(state_validation_messages[state_validation_code])

    return state_validation_code == 0, state_validation_code

if __name__ == "__main__":
    check_state_rules_for_directory(EXTRACTED_DATA_DIR)












