import pandas as pd

# %%

import numpy as np
import os
import librosa
import json
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import math
import numpy as np
from glob import glob
from IOFunctions import saveFile, loadFile, get_song_from_directory_by_identifier
import random

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'Data')
EXTRACT_DIR = os.path.join(THIS_DIR, 'DataE')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

def extract_beat_times_chroma_tempo_from_ogg(ogg_file):
    # Load sample song
    # ong_path='C:\Users\gonca\Dropbox\oxai\beatsaber\believer\Believer\song.wav'
    # data, samplerate = sf.read('existing_file.wav')
    y, fs = librosa.load(ogg_file, sr=None)

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs, onset_envelope=None, hop_length=512,
            start_bpm=120.0, tightness=100., trim=True, bpm=None, units='frames')
    """ y               : np.ndarray [shape=(n,)] or None   :: audio time series
        sr              : number > 0 [scalar]   :: sampling rate of `y`
        onset_envelope  : np.ndarray [shape=(n,)] or None   :: (optional) pre-computed onset strength envelope.
        hop_length      : int > 0 [scalar] :: number of audio samples between successive `onset_envelope` values)
        start_bpm       : float > 0 [scalar] :: initial guess for the tempo estimator (in beats per minute)
        tightness       : float [scalar] :: tightness of beat distribution around tempo
        trim            : bool [scalar] :: trim leading/trailing beats with weak onsets
        bpm             : float [scalar] :: (optional) If provided, use `bpm` as the tempo instead of estimating 
                            it from `onsets`.)
        units           : {'frames', 'samples', 'time'} :: The units to encode detected beat events in. 
                            By default, 'frames' are used.
    """
    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))]

    # Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=fs, hop_length=512, n_fft=None)
    """     
        frames      : np.ndarray [shape=(n,)] :: frame index or vector of frame indices
        sr          : number > 0 [scalar]    :: audio sampling rate
        hop_length  : int > 0 [scalar]   :: number of samples between successive frames
        n_fft       : None or int > 0 [scalar] :: Optional: length of the FFT window. If given, time conversion will 
                        include an offset of `n_fft / 2` to counteract windowing effects when using a non-centered STFT.
    """

    # %%
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=fs, C=None, hop_length=512, fmin=None,
               norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
               n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full')
    """ y               : np.ndarray [shape=(n,)] :: audio time series
        sr              : number > 0 :: sampling rate of `y`
        C               : np.ndarray [shape=(d, t)] [Optional] :: a pre-computed constant-Q spectrogram
        hop_length      : int > 0 :: number of samples between successive chroma frames
        fmin            : float > 0 :: minimum frequency to analyze in the CQT. Default: 'C1' ~= 32.7 Hz
        norm            : int > 0, +-np.inf, or None :: Column-wise normalization of the chromagram.
        threshold       : float :: Pre-normalization energy threshold.  Values below the threshold are discarded, 
                            resulting in a sparse chromagram.
        tuning          : float :: Deviation (in cents) from A440 tuning
        n_chroma        : int > 0 :: Number of chroma bins to produce
        n_octaves       : int > 0 :: Number of octaves to analyze above `fmin`
        window          : None or np.ndarray :: Optional window parameter to `filters.cq_to_chroma`
        bins_per_octave : int > 0 :: Number of bins per octave in the CQT. Default: matches `n_chroma`
        cqt_mode        : ['full', 'hybrid'] :: Constant-Q transform mode
    """

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median, pad=True, axis=-1)
    """ data        : np.ndarray :: multi-dimensional array of features
        idx         : iterable of ints or slices :: Either an ordered array of boundary indices, or an iterable collection of slice objects.
        aggregate   : function :: aggregation function (default: `np.mean`)
        pad         : boolean :: If `True`, `idx` is padded to span the full range `[0, data.shape[axis]]`
        axis        : int :: The axis along which to aggregate data
    """
    # Chop last column. Chroma features are computed between beat events  
    # Each column beat_chroma[:, k] will be the average of input columns between beat_frames[k] and beat_frames[k+1].    
    beat_chroma = beat_chroma[:, :-1]  
    return tempo, beat_times, beat_chroma

def convert_beatchroma_to_notes_position(beat_chroma):
    indexMax = np.argmax(beat_chroma, axis=0)
    num_beats = len(indexMax)
    # num_layers = 3  # rows
    nCols = 4  # columns
    # print(indexMax)

    line_layer = np.ones(indexMax.shape, dtype=int)
    line_index = np.ones(indexMax.shape, dtype=int)

    for i in range(0, num_beats):
        line_layer[i] = np.floor(indexMax[i] / nCols).astype(int)
    for i in range(0, num_beats):
        line_index[i] = np.mod(indexMax[i], nCols).astype(int)

    # print(indexMax)
    return line_layer, line_index

def convert_note_positions_to_cut_direction(line_layer, line_index):
    num_beats = len(line_layer)
    cut_direction = [8]
    #cutDirection: note cut direction (  0=up; 1=down; 2=left; 3=right;
    #                                        4=up-left; 5=up-right; 6=down-left; 7=down-right;
    #                                        8=no-direction
    reverse_directions = [1, 0, 3, 2, 7, 6, 5, 4, 8]
    #reverse_horizontal = [0, 1, 3, 2, 3, 2, 3, 2, 8]
    #reverse_vertical = [1, 0, 2, 3, 1, 1, 2, 2, 8]

    for i in range(1, num_beats):
        if line_layer[i] > line_layer[i-1]:
            #up
            if line_index[i] > line_index[i-1]:
                #right
                cut_direction.append(5)
            elif line_index[i] < line_index[i-1]:
                #left
                cut_direction.append(4)
            else:
                # no-hor change
                cut_direction.append(0)
        elif line_layer[i] < line_layer[i-1]:
            #down
            if line_index[i] > line_index[i-1]:
                #right
                cut_direction.append(7)
            elif line_index[i] < line_index[i-1]:
                #left
                cut_direction.append(6)
            else:
                # no-hor change
                cut_direction.append(1)
        else:
            #no-vert change
            if line_index[i] > line_index[i-1]:
                #right
                cut_direction.append(3)
            elif line_index[i] < line_index[i-1]:
                #left
                cut_direction.append(2)
            else:
                #no-hor change
                cut_direction.append(reverse_directions(cut_direction[-1]))
                cut_direction.append(8)


def generate_note_types_from_line_index(line_index):
    num_beats = len(line_index)
    type = []
    """type: Note type (0 = red, 1 = blue, 3 = bomb)"""
    # Assuming red is left and blue is right
    for i in range(num_beats):
        if line_index >= 2:
            type.append(1)
        else:
            type.append(0)

def filter_notes_by_patterns(line_index, line_layer, beat_times_beats, beat_duration, difficulty):
    beats_per_bar = 4  # an assumption for now
    pattern_length = 2 * beats_per_bar
    num_beats = len(line_index)
    num_types = 3

    easy_patterns = []
    easy_patterns.append([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    easy_patterns.append([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    easy_patterns.append([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    easy_patterns.append([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    easy_patterns.append([[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

    med_patterns = []
    med_patterns.append([[1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    med_patterns.append([[1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    med_patterns.append([[0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    med_patterns.append([[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    med_patterns.append([[0, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    med_patterns.append([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])

    hard_patterns = []
    hard_patterns.append([[1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0]])
    hard_patterns.append([[1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    hard_patterns.append([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0]])

    expert_patterns = []
    expert_patterns.append([[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    expert_patterns.append([[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    expert_patterns.append([[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1]])
    expert_patterns.append([[0, 0, 0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1]])
    expert_patterns.append([[1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    expert_patterns.append([[0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    expert_patterns.append([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    expert_patterns.append([[0, 1, 1, 1, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

    all_patterns = [easy_patterns, med_patterns, hard_patterns, expert_patterns]
    num_patterns = [len(easy_patterns), len(med_patterns), len(hard_patterns), len(expert_patterns)]

    #The following decides the proportion of patterns in each difficulty
    pattern_difficulty_splits = [[1, 1, 1, 1],  #easy
                                 [0.4, 1, 1, 1],  #normal
                                 [0.2, 0.6, 1, 1],  #hard
                                 [0, 0.2, 0.6, 1],  #expert
                                 [0, 0, 0.2, 1]]  #expert+

    #The following performs sampling from the patterns according to the preceding probabilities
    pattern_difficulty = np.random.rand(num_beats)

    for i in range(num_beats):  # There's a better way to do this than a for loop, but i'm lazy right now
        pattern_difficulty[i] = np.where(pattern_difficulty_splits[difficulty] > pattern_difficulty[i])[0][0]
    # pattern_difficulty = np.where(pattern_difficulty_splits[difficulty] > pattern_difficulty)[1]
    pattern_difficulty = pattern_difficulty.astype(int)

    intermediate_beat_duration = beat_duration / pattern_length
    output_line_index = []
    output_line_layer = []
    output_note_type = []
    output_beat_times = []

    for curr_beat in range(num_beats):
        these_patterns = all_patterns[pattern_difficulty[curr_beat]]
        pattern_id = np.random.randint(num_patterns[pattern_difficulty[curr_beat]], size=1)[0]
        this_pattern = these_patterns[pattern_id]
        for i in range(pattern_length):
            for j in range(num_types):
                if this_pattern[j][i] == 1:
                    if j == 0:  # red
                        if line_index[curr_beat] >= 2:  #if right switch to left
                            output_line_index.append(3 - line_index[curr_beat])
                        else:
                            output_line_index.append(line_index[curr_beat])
                    if j == 1:  # blue
                        if line_index[curr_beat] < 2:  #if left switch to right
                            output_line_index.append(3 - line_index[curr_beat])
                        else:
                            output_line_index.append(line_index[curr_beat])
                    if j == 2:  # bomb
                        if np.random.randint(0, 2, 1)[0] == 1:  #coin == heads
                            output_line_index.append(3 - line_index[curr_beat])
                        else:
                            output_line_index.append(line_index[curr_beat])
                    output_line_layer.append(line_layer[curr_beat])
                    output_note_type.append(j)
                    output_beat_times.append(beat_times_beats[curr_beat] + (intermediate_beat_duration * i))
    return output_line_layer, output_line_index, output_note_type, output_beat_times


def convert_note_positions_and_type_to_cut_direction(line_layer, line_index, note_type):
    num_beats = len(line_layer)
    cut_direction = [8]
    # cutDirection: note cut direction (  0=up; 1=down; 2=left; 3=right;
    #                                        4=up-left; 5=up-right; 6=down-left; 7=down-right;
    #                                        8=no-direction
    reverse_directions = [1, 0, 3, 2, 7, 6, 5, 4, 8]
    # reverse_horizontal = [0, 1, 3, 2, 3, 2, 3, 2, 8]
    # reverse_vertical = [1, 0, 2, 3, 1, 1, 2, 2, 8]
    last_of_type = [-1, -1, -1]
    for i in range(1, num_beats):
        if(last_of_type[note_type[i]] == -1):
            last_idx = i-1
        else:
            last_idx = last_of_type[note_type[i]]
        if line_layer[i] > line_layer[last_idx]:
            # up
            if line_index[i] > line_index[last_idx]:
                # right
                cut_direction.append(5)
            elif line_index[i] < line_index[last_idx]:
                # left
                cut_direction.append(4)
            else:
                # no-hor change
                cut_direction.append(0)
        elif line_layer[i] < line_layer[last_idx]:
            # down
            if line_index[i] > line_index[last_idx]:
                # right
                cut_direction.append(7)
            elif line_index[i] < line_index[last_idx]:
                # left
                cut_direction.append(6)
            else:
                # no-hor change
                cut_direction.append(1)
        else:
            # no-vert change
            if line_index[i] > line_index[last_idx]:
                # right
                cut_direction.append(3)
            elif line_index[i] < line_index[last_idx]:
                # left
                cut_direction.append(2)
            else:
                # no-hor change
                cut_direction.append(reverse_directions[cut_direction[-1]])
        last_of_type[note_type[i]] = i
    return cut_direction


def generate_beatsaber_notes_from_beat_times_and_chroma(beat_times, beat_chroma, tempo, difficulty):
    #beat_times_t = beat_times[:, None]
    #beat_times_T = beat_times_t.T

    # %% Convert beat times to time in beats
    beat_length = (tempo / 60)
    beat_times_beats = beat_times * beat_length
    beat_duration = np.mean(beat_times_beats[1:] - beat_times_beats[:-1])

    # %% Convert beat chroma t into line Index and Layer
    line_layer, line_index = convert_beatchroma_to_notes_position(beat_chroma)
    #note_type = generate_note_types_from_line_index(line_index)
    line_layer, line_index, note_type, beat_times_beats = \
        filter_notes_by_patterns(line_index, line_layer, beat_times_beats, beat_duration, difficulty)
    cut_direction = convert_note_positions_and_type_to_cut_direction(line_layer, line_index, note_type)

    num_beats = len(note_type)
    # %%
    #beat_features = np.vstack([beat_times_T, beat_chroma_chopped])

    # $$
    # %% Create Data frames
    dict = {'_cutDirection': [], '_lineIndex': [], '_lineLayer': [], '_time': [], '_type': []}
    notes = pd.DataFrame.from_dict(dict)

    # print(np.shape(beat_times_beats))

    for i in range(0, num_beats):
        new_note = {'_cutDirection': [cut_direction[i]], '_lineIndex': [line_index[i]], '_lineLayer': [line_layer[i]], '_time': [beat_times_beats[i]], '_type': [note_type[i]]}
        df_new = pd.DataFrame.from_dict(new_note)
        notes = notes.append(df_new)

    return notes

def baseline_notes_simple():
    dict = {'_cutDirection': [], '_lineIndex': [], '_lineLayer': [], '_time': [], '_type': []}
    notes_random = pd.DataFrame.from_dict(dict)

    for i in range(0,418):
        new_note = {'_cutDirection': [0], '_lineIndex': [1], '_lineLayer': [1], '_time': [i], '_type': [1]}
        df_new = pd.DataFrame.from_dict(new_note)
        notes_random = notes_random.append(df_new)
    return notes_random

# beat_times: Time (in seconds) of each given frame number
def generate_beatsaber_obstacles_from_beat_times (beat_times, tempo, difficulty):
    numObstacles = 0
    #depending on difficulty the number of obstacles can increase, we can change this of course
    if difficulty == 1: # Normal
        numObstacles = 5
    elif difficulty == 2: # Hard
        numObstacles = 15
    elif difficulty == 3: # Expert
        numObstacles = 30
    elif difficulty == 4: # ExpertPlus
        numObstacles = 60
    #_, beat_times, _ = extract_beat_times_chroma_tempo_from_ogg(ogg_file) # we only want the beat_times here
    time = []
    lineIndex = []
    type = []
    duration = []
    width = []

    # create data frames
    dict = {'_time': [], '_lineIndex': [], '_type': [], '_duration': [], '_width': []}
    obstacles = pd.DataFrame.from_dict(dict)

    # where we create our new obstacle components and obstacles
    for i in range(numObstacles):
        #setting the values of our obstacle components
        randomTime = random.choice(beat_times)
        while randomTime in time:
            randomTime = random.choice(beat_times)
        time.append(randomTime)
        lineIndex.append(random.randint(0, 3))
        type.append(random.randint(0,1))
        duration.append(random.randint(1, 3))
        if(lineIndex[i] == 0 or lineIndex[i] == 1):
            width.append(randint(1, 3))
        elif(lineIndex[i] == 2):
            width.append(random.randint(1, 2))
        elif(lineIndex[i] == 3):
            width.append(1)

        # creating our new obstacle
        new_obstacle = {'_time': [time[i]], '_lineIndex': [lineIndex[i]], '_type': [type[i]], '_duration': [duration[i]], '_width': [width[i]]}
        df_new = pd.DataFrame.from_dict(new_obstacle)
        obstacles = obstacles.append(df_new)

    return obstacles # time, lineIndex, type, duration, width


def generate_beatsaber_notes_from_ogg(ogg_file, difficulty=0):
    meta_dir = os.path.dirname(ogg_file)
    meta_filename = 'meta_info.pkl'
    meta_file = os.path.join(meta_dir, meta_filename)
    if os.path.isfile(meta_file):
        content = loadFile(meta_file)
        tempo = content[0]
        beat_times = content[1]
        beat_chroma = content[2]
    else:
        tempo, beat_times, beat_chroma = extract_beat_times_chroma_tempo_from_ogg(ogg_file)
        saveFile([tempo, beat_times, beat_chroma], meta_filename, meta_dir, append=False)
    notes = generate_beatsaber_notes_from_beat_times_and_chroma(beat_times, beat_chroma, tempo, difficulty)
    return notes

def generate_beatsaber_obstacles_from_ogg(ogg_file, difficulty=0):
    meta_dir = os.path.dirname(ogg_file)
    meta_filename = 'meta_info.pkl'
    meta_file = os.path.join(meta_dir, meta_filename)
    if os.path.isfile(meta_file):
        content = loadFile(meta_file)
        tempo = content[0]
        beat_times = content[1]
        beat_chroma = content[2]
    else:
        tempo, beat_times, beat_chroma = extract_beat_times_chroma_tempo_from_ogg(ogg_file)
        saveFile([tempo, beat_times, beat_chroma], meta_filename, meta_dir, append=False)
    obstacles = generate_beatsaber_obstacles_from_beat_times(beat_times, tempo, difficulty)
    return obstacles

if __name__ == '__main__':
    song_directory, song_ogg, song_json, song_filename = get_song_from_directory_by_identifier('believer')
    notes = generate_beatsaber_notes_from_ogg(song_ogg)
    obstacles = generate_beatsaber_obstacles_from_ogg(song_ogg)