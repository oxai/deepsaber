import IOFunctions
from identifyStateSpace import compute_explicit_states_from_json
import math, numpy as np
import librosa
import os
#from featuresBase import extract_beat_times_chroma_tempo_from_ogg

'''
This file contains all helper functions to take a JSON level file and convert it to the current note representation
Requirements: The stateSpace directory. It contains sorted_states.pkl, which stores all identified states in the dataset.
To generate this folder, run identifyStateSpace.py
'''
NUM_DISTINCT_STATES = 4672 # This is the number of distinct states in our dataset
EMPTY_STATE_INDEX = 0 # or NUM_DISTINCT_STATES. CONVENTION: The empty state is the zero-th state.
SAMPLING_RATE = 16000
# NUM_SPECIAL_STATES=2 # also padding

def compute_state_sequence_representation_from_json(json_file, states=None, top_k=2000):
    '''

    :param json_file: The input JSON level file
    :param top_k: the top K states to keep (discard the rest)
    :return: The sequence of state ranks (of those in the top K) appearing in the level
    '''
    if states is None:  # Only load if state is not passed
        states = IOFunctions.loadFile("sorted_states.pkl", "stateSpace") # Load the state representation
    if EMPTY_STATE_INDEX == 0:  # RANK 0 is reserved for the empty state
        # states_rank = {state: i+1 for i, state in enumerate(states)}
        states_rank = {state: i+1 for i, state in enumerate(states)}
    else: # The empty state has rank NUM_DISTINCT_STATES
        states_rank = {state: i for i, state in enumerate(states)}
    explicit_states = compute_explicit_states_from_json(json_file)
    # Now map the states to their ranks (subject to rank being below top_k)
    state_sequence = {time: states_rank[exp_state] for time, exp_state in explicit_states.items()
                      if states_rank[exp_state] <= top_k}
    return state_sequence


def get_block_sequence_with_deltas(json_file, song_length, bpm, top_k=2000, beat_discretization = 1/16,states=None,one_hot=False):
    state_sequence = compute_state_sequence_representation_from_json(json_file=json_file, top_k=top_k, states=states)
    times_beats = np.array([time for time, state in state_sequence.items() if (time*60/bpm) <= song_length])
    feature_indices = np.array([int((time/beat_discretization)+0.5) for time in times_beats])  # + 0.5 is for rounding
    times_real = times_beats * (60/bpm)
    states = np.array([state for time, state in state_sequence.items() if (time*60/bpm) <= song_length])
    pos_enc = np.arange(len(states))
    if one_hot:
        one_hot_states = np.zeros((top_k + 1, states.shape[0]))
        one_hot_states[states, pos_enc] = 1
    time_diffs = np.diff(times_real)
    delta_backward = np.expand_dims(np.insert(time_diffs, 0, times_real[0]), axis=0)
    delta_forward = np.expand_dims(np.append(time_diffs, song_length - times_real[-1]), axis=0)
    if one_hot:
        return one_hot_states, pos_enc, delta_forward, delta_backward, feature_indices
    else:
        return states, pos_enc, delta_forward, delta_backward, feature_indices



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
                                        top_k=top_k,beat_discretization=beat_discretization, audio_feature_select=True)
        break
    return directory_dict    #TODO: Add some code here to save the representations eventually


def extract_representations_from_song_directory(directory,top_k=2000,beat_discretization=1/16, audio_feature_select="Hybrid"):
    OGG_files = IOFunctions.get_all_ogg_files_from_data_directory(directory)
    if len(OGG_files) == 0:  # No OGG file ... skip
        print("No OGG file for song "+directory)
        return
    OGG_file = OGG_files[0]  # There should only be one OGG file in every directory anyway, so we get that
    JSON_files = IOFunctions.get_all_json_level_files_from_data_directory(directory)
    if len(JSON_files) == 0:  # No Non-Autosave JSON files
        JSON_files = IOFunctions.get_all_json_level_files_from_data_directory(directory, include_autosaves=True)
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
        bs_level = IOFunctions.parse_json(JSON_file)
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
            audio_features = chroma_feature_extraction(y, sr, feature_extraction_times)
        elif audio_feature_select == "MFCC":  # for mfccs
            audio_features = mfcc_feature_extraction(y, sr, feature_extraction_times)
        elif audio_feature_select == "Hybrid":
            audio_features = feature_extraction_hybrid(y, sr, feature_extraction_times,bpm,beat_discretization)
        # chroma_features = chroma_feature_extraction(y,sr, feature_extraction_frames, bpm, beat_discretization)
        level_state_feature_maps[os.path.basename(JSON_file)] = (level_states, audio_features)

        # To obtain the chroma features for each pitch you access it like: chroma_features[0][0]
        # the first index number refers to the 12 pitches, so is indexes 0 to 11
        # the second index number refers to the chroma values, so is indexed from 0 to numOfStates - 1
        # print(audio_features.shape) # Sanity Check

        # WE SHOULD ALSO USE THE PERCUSSIVE FREQUENCIES IN OUR DATA, Otherwise the ML is losing valuable information
    return level_state_feature_maps


def feature_extraction_hybrid_raw(y,sr,bpm,beat_discretization=1/16,mel_dim=12,window_mult=1):
    beat_duration = int(60 * sr / bpm)  # beat duration in samples
    hop = int(beat_duration * beat_discretization) # one vec of mfcc features per 16th of a beat (hop is in num of samples)
    hop -= hop % 32
    window = window_mult * hop
    y_harm, y_perc = librosa.effects.hpss(y)
    mels = librosa.feature.melspectrogram(y=y_perc, sr=sr,n_fft=window,hop_length=hop,n_mels=mel_dim, fmax=65.4)  # C2 is 65.4 Hz
    cqts = librosa.feature.chroma_cqt(y=y_harm, sr=sr,hop_length= hop,
                                      norm=np.inf, threshold=0, n_chroma=12,
                                      n_octaves=6, fmin=65.4, cqt_mode='full')
    joint = np.concatenate((mels, cqts), axis=0)
    return joint


def feature_extraction_hybrid(y, sr, state_times,bpm,beat_discretization=1/16,mel_dim=12):
    y_harm, y_perc = librosa.effects.hpss(y)
    hop = 256 # Tnis is the default hop length
    SANITY_RATIO = 0.25 # HAS TO BE AT MOST  0.5 to produce different samples per beat
    if hop > SANITY_RATIO * beat_discretization * sr * (60 / bpm):
        hop = int(SANITY_RATIO * beat_discretization * sr * 60 / bpm) # Make small enough to do the job. NOTE: FIX ME
        hop -= hop % 32 # Has to be a multiple of 32 for CQT to work
        if hop <= 0:
            hop = 32 # Just in Case
        # print(hop) # Sanity Check
    mels = librosa.feature.melspectrogram(y=y_perc, sr=sr, n_mels=mel_dim, fmax=65.4, hop_length=hop)  # C2 is 65.4 Hz
    cqts = librosa.feature.chroma_cqt(y=y_harm, sr=sr, C=None, hop_length=hop,
                                      norm=np.inf, threshold=0, tuning=None, n_chroma=12,
                                      n_octaves=6, fmin=65.4, window=None, cqt_mode='full')
    # Problem: Sync is returning shorter sequences than the state times
    state_frames = librosa.core.time_to_frames(state_times,hop_length=hop,sr=sr) # Hop-Aware Synchronisation
    # print(state_frames)
    beat_chroma = librosa.util.sync(cqts, state_frames, aggregate=np.median, pad=True, axis=-1)
    beat_mel = librosa.util.sync(mels, state_frames, aggregate=np.median,pad=True,axis=-1)
    output = np.concatenate((beat_mel,beat_chroma), axis=0)
    return output

def feature_extraction_mel(y, sr, state_times,bpm,beat_discretization=1/16,mel_dim=100):
    # y_harm, y_perc = librosa.effects.hpss(y)
    hop = 256 # Tnis is the default hop length
    SANITY_RATIO = 0.25 # HAS TO BE AT MOST  0.5 to produce different samples per beat
    if hop > SANITY_RATIO * beat_discretization * sr * (60 / bpm):
        hop = int(SANITY_RATIO * beat_discretization * sr * 60 / bpm) # Make small enough to do the job. NOTE: FIX ME
        hop -= hop % 32 # Has to be a multiple of 32 for CQT to work
        if hop <= 0:
            hop = 32 # Just in Case
        # print(hop) # Sanity Check
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_dim, hop_length=hop)  # C2 is 65.4 Hz
    # Problem: Sync is returning shorter sequences than the state times
    state_frames = librosa.core.time_to_frames(state_times,hop_length=hop,sr=sr) # Hop-Aware Synchronisation
    # print(state_frames)
    beat_chroma = librosa.util.sync(cqts, state_frames, aggregate=np.median, pad=True, axis=-1)
    beat_mel = librosa.util.sync(mels, state_frames, aggregate=np.median,pad=True,axis=-1)
    output = np.concatenate((beat_mel,beat_chroma), axis=0)
    return output

def chroma_feature_extraction(y,sr, state_times):
    #hop = #int((44100 * 60 * beat_discretization) / bpm) Hop length must be a multiple of 2^6
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, C=None, fmin=None,
                                            norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
                                            n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full')
    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    state_frames = librosa.core.time_to_frames(state_times,sr=sr) # Default hop length of 512
    #TODO: CHANGE THIS TO BECOME LIKE HYBRID IF WE ARE TO EVER USE THIS
    beat_chroma = librosa.util.sync(chromagram, state_frames, aggregate=np.median, pad=True, axis=-1)
    return beat_chroma


def mfcc_feature_extraction(y,sr,state_times):
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # we can add other specified parameters
    state_frames = librosa.core.time_to_frames(state_times,sr=sr)
    beat_mfcc = librosa.util.sync(mfcc, state_frames, aggregate=np.median, pad=True, axis=-1)
    return beat_mfcc
