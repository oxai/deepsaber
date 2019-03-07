import IOFunctions
from identifyStateSpace import compute_explicit_states_from_json
import math, numpy as np
from featuresBase import extract_beat_times_chroma_tempo_from_ogg
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

def chroma_feature_extraction(ogg_file, json_file, beat_discretization = 1/16):
    # Load sample song
    y, fs = librosa.load(ogg_file, sr=None)

    bsLevel = parse_json(json_file)
    bpm = bsLevel["_beatsPerMinute"]
    hop = int((44100 * 60 * (beat_discretization)) / bpm)
    # hop_length      : int > 0 :: number of samples between successive chroma frames

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs, onset_envelope=None, hop_length=hop,
                                                 start_bpm=bpm, tightness=100., trim=True, bpm=None, units='frames')
    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))]

    # Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=fs, hop_length=hop, n_fft=None)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=fs, C=None, hop_length=hop, fmin=None,
                                            norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
                                            n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full')

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median, pad=True, axis=-1)

    # Chop last column. Chroma features are computed between beat events
    # Each column beat_chroma[:, k] will be the average of input columns between beat_frames[k] and beat_frames[k+1].
    beat_chroma = beat_chroma[:, :-1]

    return beat_chroma





