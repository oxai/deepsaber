import librosa, os, IOFunctions
import soundfile as sf, numpy as np
DATA_DIRECTORY = "Data/"


def pitch_shift(ogg_file, nb_half_tones, out_dir):
    y,sr = librosa.load(ogg_file)
    y_shifted = librosa.effects.pitch_shift(y, sr, nb_half_tones, bins_per_octave=12)
    sf.write(out_dir, y_shifted, sr, format='ogg', subtype='vorbis')

def add_noise(data, level):
    noise = np.random.randn(len(data))
    data_noisy = data + level * noise
    return data_noisy

def noise_insertion(ogg_file, noise_level, out_dir):
    y, sr = librosa.load(ogg_file)
    y_noisy = add_noise(y, noise_level)
    sf.write(out_dir, y_noisy, sr, format='ogg', subtype='vorbis')


SHIFT_IDENTIFIER = "SHIFTEDn"
NOISE_IDENTIFIER = "NOISEDn"
SHIFT = (pitch_shift,SHIFT_IDENTIFIER)
NOISE_ADD = (noise_insertion, NOISE_IDENTIFIER)


def data_augment(augmentation_function, max_nb_half_tones = 3, nb_noise_additions = 5,noise_level = 0.005):
    # Step 1: Identify All song directories that aren't shifted to begin with
    song_directories = [os.path.join(DATA_DIRECTORY, song_dir) for song_dir in os.listdir(DATA_DIRECTORY)
                        if os.path.isdir(os.path.join(DATA_DIRECTORY, song_dir)) and augmentation_function[1]
                        not in song_dir]

    # Step 2: Augment all eligible songs in directories
    for directory in song_directories:
        print(directory)
        directory_name = os.path.basename(directory)
        # Copy Everything but the OGG file
        loop_range = 2*max_nb_half_tones if augmentation_function[1] == SHIFT_IDENTIFIER else nb_noise_additions

        for index in range(loop_range):
            new_directory_path = directory + "-"+ augmentation_function[1] + str(index+1) # Create New Directory
            new_directory_name = os.path.basename(new_directory_path)
            if not os.path.exists(new_directory_path):
                os.makedirs(new_directory_path)
                print(new_directory_path)
                # Now copy all non-OGG files
                command_string = "rsync -av --exclude='*.ogg' \""+directory+"\"/* \""+new_directory_path+"\""
                os.system(command_string)
                # Now fetch all OGG files
                ogg_files = IOFunctions.get_all_ogg_files_from_data_directory(directory)
                new_ogg_files = [ogg.replace(directory_name,new_directory_name, 1) for ogg in ogg_files]
                for i, ogg in enumerate(ogg_files):
                    if augmentation_function[1] == SHIFT_IDENTIFIER: # SHIFTING AUGMENTATION
                        shift_quantity = (int(index/2) + 1) * (1 - 2*(index % 2)) # 0 -> 1, 1 -> -1, 2 -> 2, 3 -> -2,etc
                        print(shift_quantity)
                        augmentation_function[0](ogg, shift_quantity, new_ogg_files[i])
                    else: # NOISE ADDITION
                        augmentation_function[0](ogg, noise_level, new_ogg_files[i])
                print(ogg_files)
                print(new_ogg_files)
            else:
                continue # Only run augmentation if it hasn't run already run before


if __name__ == "__main__":
    data_augment(NOISE_ADD, noise_level= 0.005) # For noise insertion
    data_augment(SHIFT, max_nb_half_tones= 3) # For pitch shifting


