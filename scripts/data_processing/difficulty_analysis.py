import getpass
import time
import os
import numpy as np

from scripts.misc import io_functions
from scripts.misc.io_functions import read_meta_data_file, get_list_of_downloaded_songs, get_all_json_level_files_from_data_directory
from scripts.data_processing.state_space_functions import compute_explicit_states_from_bs_level

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.pardir(os.pardir(THIS_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

''' @TS: Trying to regress a model of difficulty from beatsaber level features:
x_00) Number of blocks
x_01) Number of obstacles
x_02) Number of bombs
x_04) Number of unique states (bombs + blocks)
x_04) Beats per minute
x_05) Blocks per minute
x_06) Blocks per beat
x_07) Song length
x_08) Distance Travelled
x_09) Angles Travelled
x_10) Product Distance Travelled

y_0) Difficulty rating

z_0) Scoresaber difficulty rating

_Version 1_
Linear model: y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
'''

def extract_features_from_all_levels():
    downloaded_songs_full, downloaded_songs = get_list_of_downloaded_songs()

    features = np.array([])
    targets = np.array([])
    features_needed = []
    for song_dir in downloaded_songs_full:
        features_needed.append(song_dir)
    from multiprocessing import Pool
    p = Pool(8)
    p.map(extract_features_targets_from_dir, features_needed)

    print('Reading features_and_targets from song dirs')
    for song_dir in downloaded_songs_full:
        feature_target = read_features_targets_from_song_dir(song_dir)
        if feature_target is not None and len(feature_target) == 2:
            if len(feature_target[0]) != 0 and len(feature_target[1]) != 0:
                features = np.append(features, feature_target[0])
                targets = np.append(targets, feature_target[1])
    features_and_targets = np.stack([features, targets], axis=1)
    io_functions.saveFile(features_and_targets, os.path.join(EXTRACT_DIR, 'features_and_targets_' + getpass.getuser() + '_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.pkl'))

    return features_and_targets


def extract_features_targets_from_dir(song_dir):
    print('Extracting Features from '+str(song_dir))
    meta_data_filename = os.path.join(EXTRACT_DIR, os.path.join(song_dir, 'meta_data.txt'))
    features = np.array([])
    targets = np.array([])
    if not os.path.exists(meta_data_filename):
        pass
    else:
        meta_data = read_meta_data_file(meta_data_filename)
        difficulty_rating = meta_data['scoresaberDifficulty'].replace(' ', '').split(',')
        try:
            if difficulty_rating != ['']:
                json_files = get_all_json_level_files_from_data_directory(os.path.join(EXTRACT_DIR, song_dir))
                if(len(json_files) == len(difficulty_rating)):
                    for i in range(len(json_files)):
                        bs_level = io_functions.parse_json(json_files[i])
                        features = np.append(features, np.array(extract_features_from_beatsaber_level(bs_level)))
                        try:
                            targets = np.append(targets, np.array(
                                [float(difficulty_rating[i]), int(meta_data['thumbsUp']), int(meta_data['thumbsDown']),
                                 float(meta_data['rating']), \
                                 float(meta_data['funFactor']), float(meta_data['rhythm']), float(meta_data['flow']), \
                                 float(meta_data['patternQuality']), float(meta_data['readability']),
                                 float(meta_data['levelQuality'])]))
                        except IndexError:
                            print(difficulty_rating)
                            print(i)
                            print(meta_data)
            if len(features) is not 0 and len(targets) is not 0:
                features_and_targets = np.stack([features, targets], axis=1)
                io_functions.saveFile(features_and_targets, os.path.join(EXTRACT_DIR, os.path.join(song_dir, 'features_targets.pkl')))
        except Exception:
            print(Exception)
    return [features, targets]

def read_features_targets_from_song_dir(song_dir):
    features_targets_filename = os.path.join(EXTRACT_DIR, os.path.join(song_dir, 'features_targets.pkl'))
    if not os.path.exists(features_targets_filename):
        features_and_targets = None
    else:
        features_and_targets = np.array(io_functions.loadFile(features_targets_filename))
    return features_and_targets

def extract_features_from_beatsaber_level(bs_level):
    #feature_1 = distance travelled
    #feature_2 = angles travelled
    #feature_3 = sum of product distance/angle vectors

    num_blocks = extract_level_num_blocks(bs_level)
    num_obstacles = extract_level_num_obstacles(bs_level)
    num_bombs = extract_level_num_bombs(bs_level)
    num_unique_states = extract_level_num_unique_states(bs_level)

    beats_per_minute = extract_level_beats_per_minute(bs_level)
    blocks_per_minute = extract_level_blocks_per_minute(bs_level)
    blocks_per_beat = extract_level_blocks_per_beat(bs_level)
    song_length = extract_level_song_length(bs_level)

    [distance_acc_blue, distance_acc_red, velocity_blue, velocity_red] = extract_level_distance_velocity(bs_level)
    [angles_travelled_blue, angles_travelled_red] = extract_level_angles_travelled(bs_level)
    product_distance_travelled = extract_level_product_distance_travelled(bs_level)

    return [num_blocks, num_obstacles, num_bombs, num_unique_states,\
           beats_per_minute, blocks_per_minute, blocks_per_beat, song_length, \
           distance_acc_blue, distance_acc_red, velocity_blue, velocity_red, \
           angles_travelled_blue, angles_travelled_red, product_distance_travelled]


def extract_level_num_blocks(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    num_blocks = 0
    for entry in notes.itertuples():
        if entry[5] != 3:  # This is not a bomb
            num_blocks += 1
    return num_blocks


def extract_level_num_obstacles(bs_level):
    return bs_level['_obstacles'].shape[0]


def extract_level_num_bombs(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    num_bombs = 0
    for entry in notes.itertuples():
        if entry[5] == 3:  # This is a bomb
            num_bombs += 1
    return num_bombs


def extract_level_num_unique_states(bs_level):
    return len(compute_explicit_states_from_bs_level(bs_level))


def extract_level_beats_per_minute(bs_level):
    return bs_level['_beatsPerMinute']


def extract_level_blocks_per_minute(bs_level):
    return extract_level_num_blocks(bs_level) / extract_level_song_length(bs_level)


def extract_level_blocks_per_beat(bs_level):
    return extract_level_blocks_per_minute(bs_level) / extract_level_beats_per_minute(bs_level)


def extract_level_song_length(bs_level):
    notes = bs_level["_notes"]  # Parse the JSON notes to use the notes representation
    note_times = set(notes["_time"])
    return max(note_times)


def extract_level_distance_velocity(bs_level):
    #input is a json file with beatsaber data, json path
    #event Type is either: "events", "notes", "obstacles"

    array_blue, array_red = extract_notes_from_bs_level(bs_level)

    # event value = 0 or 1 (red and blue)
    [distance_acc_blue, velocity_blue] = return_distance_velocity(array_blue)
    [distance_acc_red, velocity_red] = return_distance_velocity(array_red)
    return distance_acc_blue, distance_acc_red, velocity_blue, velocity_red


def return_distance_velocity(df):
    #initialize starting positions
    tMinus1 = 0
    rowMinus1 = 0
    columnMinus1 = 0
    distance_acc = 0
    counter = 0
    t = 0
    for index, element in df.iterrows():
        counter += 1
        t = element['_time']
        row = element['_lineIndex']
        column = element['_lineLayer']
        distance = ((row-rowMinus1)**2+(column-columnMinus1)**2)**(1/2)
        dt = t-tMinus1
        if dt ==0: #because there are some blocks postioned right next to each other at the same time.
            dt = 0.1
        #velocity = distance/(dt)*bpm #convert to blocks per beat to blocks per second
        #velocity_avg = distance_acc/t*bpm
        rowMinus1 = row
        columnMinus1 = column
        tMinus1 = t
        distance_acc += distance

    if t != 0:
        velocity_avg = distance_acc/t
    else:
        velocity_avg = 0

    return distance_acc, velocity_avg


# v1.v2 = |V1||V2|cos(theta)
# cos(theta) = v1.v2 / |V1||V2|
# sin(theta) = sqrt(1 - cos(theta))
# outer angle is angle of difficulty / 180 - cos(angle)


def calc_angle_of_vector(vector):
    #vector = [dx, dy]
    #angle = arctan(dx/dy)
    assert(len(vector) == 2)
    if vector[1] == 0:
        if vector[0] < 0:
            angle = np.arctan(-np.inf)
        elif vector[0] > 0:
            angle = np.arctan(np.inf)
        else:
            angle = None
    elif vector[0] == 0:
        if vector[1] < 0:
            angle = np.pi
        if vector[1] > 0:
            angle = 0
        else:
            angle = None
    else:
        angle = np.arctan(vector[0] / vector[1])
    return angle

def calc_vector_of_points(pt1, pt2):
    return [pt2[0] - pt1[0], pt2[1] - pt1[1]]

def extract_notes_from_bs_level(bs_level):
    df_notes = bs_level['_notes']

    # separate the red and blue blocks
    array_blue = df_notes.loc[df_notes['_type'] == 0]
    array_red = df_notes.loc[df_notes['_type'] == 1]
    return array_blue, array_red

def calc_angles_travelled(df):
    angles_travelled = 0
    i = 0
    # while df.iloc[i]['_time'] == df.iloc[i+1]['_time']:
    #     pass
    #  i+=1

    cut_direction_delta_switcher = [
        [[0., -0.5], [0., 0.5]], #up cut
        [[0., 0.5], [0., -0.5]], #down cut
        [[0.5, 0.], [-0.5, 0.]], #left cut
        [[-0.5, 0.], [0.5, 0.]], #right cut
        [[0.5, -0.5], [-0.5, 0.5]], #up-left cut
        [[-0.5, -0.5], [5.0, 0.5]], #up-right cut
        [[0.5, 0.5], [-0.5, -0.5]], #down-left cut
        [[-0.5, 0.5], [0.5, -0.5]], #down-right cut
        [[0., 0.], [0., 0.]],  #no direction
    ]

    if len(df) > 0:
        last_angle = None
        #pt1 and pt2 are created from the first note position plus start and end points, based on the cut direction
        pt1 = [df.iloc[i]['_lineIndex'], df.iloc[i]['_lineLayer']]
        ptc1 = np.add(pt1, cut_direction_delta_switcher[int(df.iloc[i]['_cutDirection'])][0])
        ptc2 = np.add(pt1, cut_direction_delta_switcher[int(df.iloc[i]['_cutDirection'])][1])
        for i in range(1, len(df) - 1, 1):
            # pt3 and pt4 are created from subsequent note positions plus start and end points
            pt2 = [df.iloc[i]['_lineIndex'], df.iloc[i]['_lineLayer']]
            ptc3 = np.add(pt2, cut_direction_delta_switcher[int(df.iloc[i]['_cutDirection'])][0])
            ptc4 = np.add(pt2, cut_direction_delta_switcher[int(df.iloc[i]['_cutDirection'])][1])
            #calc angle between beginning of note[i-1] and end of note[i-1]
            vec = calc_vector_of_points(ptc1, ptc2)
            angle = calc_angle_of_vector(vec)
            #If the two points are in the same location then the angle between them is none
            if angle is not None:
                if last_angle is None:
                    last_angle = angle # first angle that != none
                else:
                    angle_diff = np.abs(angle - last_angle)
                    angles_travelled += angle_diff
                    last_angle = angle

            # calc angle between end of note[i-1] and beginning of note[i]
            vec = calc_vector_of_points(ptc2, ptc3)
            angle = calc_angle_of_vector(vec)
            if angle is not None:
                if last_angle is None:
                    last_angle = angle
                else:
                    angle_diff = np.abs(angle - last_angle)
                    angles_travelled += angle_diff
                    last_angle = angle
            else:
                angles_travelled += np.pi #  start and end point in same position, reversal of direction is necessary
            #update notes[i-1] while 0 < i < len(df)
            pt1 = pt2
            ptc1 = ptc3
            ptc2 = ptc4
    return angles_travelled


def extract_level_angles_travelled(bs_level):
    array_blue, array_red = extract_notes_from_bs_level(bs_level)

    angles_travelled_blue = calc_angles_travelled(array_blue)
    angles_travelled_red = calc_angles_travelled(array_red)

    return [angles_travelled_blue, angles_travelled_red]


def extract_level_product_distance_travelled(bs_level):
    return 0


#actually not really need...oh well
def convert_lin_col_to_coordinates(lineIndex, lineLayer):
    # takes as input the line and the column of the element, and returns an array 3x4, with 1 for the entry, where the element
    noteArray = np.zeros((3,4))
    for j in range(0,4):
        for i in range(0,3):
            if i == lineIndex and j == lineLayer:
                noteArray[i, j] = 1
    return noteArray

#def generate_difficulty_features():
    # Generating Data Frame
    # Step 1 - Load json file
    # Read:     Number of blocks
    #           Number of unique blocks
    #           Number of obstacles
    #           Number of bombs

    # Step 2 - Load ogg file
    # Read:     Beats per minute
    #           Song length

    # Step 3 - Load difficulty rating
    # Read:     Difficulty Rating
    #           Number of votes

def linear_regression_model(features, target):
    assert(len(features) == len(target))
    x_mean = np.mean(features, axis=0)
    y_mean = np.mean(target)
    xy_cov = np.inner(np.subtract(features, x_mean).T, np.subtract(y_mean, target))
    x_var = np.mean(np.power(np.subtract(features, x_mean), 2), axis=0)
    beta = np.divide(np.sum(xy_cov), np.sum(x_var))
    alpha = np.subtract(y_mean, np.multiply(beta, x_mean))
    return [alpha, beta]

def get_linear_regression_model_for_all_targets(features, targets):
    models = []
    num_targets = len(targets[0])
    for i in range(num_targets):
        # targets indexing not compatible with slicing - ndarrayitemscontainer
        models.append(linear_regression_model(features[:], targets[:, i]))
    return models

def measure_regression_prediction_error(model, x_test, y_test):
    errors = []
    for i in range(len(x_test)):
        y_pred = np.multiply(x_test[i][:], model[1, :]).append(model[0])
        errors.append(y_pred - y_test)
    error_mean = np.mean(errors)
    error_std_dev = np.sqrt(np.mean(np.power(np.subtract(errors, error_mean), 2)))
    return errors, error_mean, error_std_dev, y_test, y_pred



if __name__ == '__main__':
    # feature_targets = extract_features_targets_from_dir('994)Made In Love - Chart by Mystikmol')
    features_and_targets = extract_features_from_all_levels()
    io_functions.saveFile(features_and_targets, 'dataset_features_and_target_metrics.pkl')
    features_and_targets = io_functions.loadFile('dataset_features_and_target_metrics.pkl')
    # models = get_linear_regression_model_for_all_targets(features_and_targets[0], features_and_targets[1])
    # io_functions.saveFile(models, 'dataset_targets_linear_model.pkl')
    # models = io_functions.loadFile('dataset_targets_linear_model.pkl')

    errors = []
    error_mean = []
    error_std_dev = []
    y_test = []
    y_pred = []
    ax = None
    new_features_and_targets = features_and_targets.copy()
    for i in range(20):
        np.random.shuffle(new_features_and_targets)
        num_samples = len(features_and_targets)
        num_train = int(np.floor(0.6*num_samples))
        num_test = num_samples - num_train
        test_models = get_linear_regression_model_for_all_targets(features_and_targets[:num_train,0], features_and_targets[:num_train,1])
        for j in range(len(test_models)):
            if j == 0 and i > 0:
                errors.append([])
                error_mean.append([])
                error_std_dev.append([])
                y_test.append([])
                y_pred.append([])
            this_errors, this_error_mean, this_error_std_dev, this_y_test, this_y_pred = measure_regression_prediction_error(test_models[j], features_and_targets[num_train:,0], features_and_targets[num_train:,1])
            errors[-1].append(this_errors)
            error_mean[-1].append(this_error_mean)
            error_std_dev[-1].append(this_error_std_dev)
            y_test[-1].append(this_y_test)
            y_pred[-1].append(this_y_pred)
            if j == 0:
                sort_idx = np.argsort(this_y_test)
                ax = io_functions.add_data_to_plot(x=y_test[-1][sort_idx], y=errors[-1][sort_idx], title='test_model_' + str(j),
                                                   ax=ax, style='r-', label=str(j), legend=True, realtime=True)

    difficulty_error = []
    difficulty_error_mean = []
    difficulty_error_std_dev = []
    for i in range(20):
        difficulty_error.append(errors[i][0])
        difficulty_error_mean.append(error_mean[i][0])
        difficulty_error_std_dev.append(error_std_dev[i][0])
        print(np.mean(difficulty_error_mean))
        print(np.mean(difficulty_error_std_dev))


