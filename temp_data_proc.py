import torch
from stateSpaceFunctions import extract_representations_from_song_directory
import numpy as np

#incorporate with Guillermo and Andrea's training code (loss, etc)
def prepare_sequence(directory):
    # the below map is indexed by os.path.basename(JSON_file)
    # the boolean in the parameters corresponds to the audio features extracted: true = chroma, false = mfcc
    state_features_map = extract_representations_from_song_directory(directory,top_k=2000,beat_discretization=1/16,
                                                                     audio_feature_select=True)
    state_features_arr = np.asarray(state_features_map)
    return torch.from_numpy(state_features_arr)