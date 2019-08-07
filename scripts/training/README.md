
Training is done by running `python3 train.py` with the relevant command line *parameters*. These parameters can be found by running `python3 train.py --help`, and are defined in the code either in the dataset class used (in `/scripts/training/data`), in the model (in `/models`) or in `/scripts/training/options/`.

Because there are quite a lot of parameters, we provide script to run standard training processes.

> To train the full DeepSaber model, run
> `./script_block_placement`
> `./script_block_selection`


#Dataset options


- `data_dir` - the directory where the extracted level/song data is found.
- `dataset_name` - the dataset class to be used (we are mostly `general_beat_saber` as of now; but stage two uses `stage_two` dataset).
- `binarized` - whether to give as targets for the model {that there exists a set of notes at this time or not} (binarized), or {the particular state at this time} (not binarized). The first is used for block placement. The second for block selection.
- `reduced_state` - whether to represent states using the reduced state representation (a class from 1 up to number_reduced_states); or as a tensor of size 12 with a class from 1 to 20 in each position, representing the whole state. The second is only used for the adv_wavenet, a GAN model, which however needs testing with the new dataset... For now use the reduced_state always.
- `feature_name` - the feature type to use. Must have run `process_songs.py` with this feature name before on the data to be used.
- `feature_size` - the feature size to use. Must have run `process_songs.py` with this feature size before on the data to be used.
- `level_diff` - a comma separated (no spaces) list of level difficulties to use in training. Should have processed them all with `process_songs.py` before.

#Model options

<small>Some models need these to be set, and others don't, depends on how general they are designed to be.</small>

- `output_channels` - When using general_beat_saber should be set to {the number of classes to predict} + 4 (the number of _special states_). When using `binarized`, the number of classes to predict is 1. Otherwise, if using `reduced_state`, it would be `number_reduced_states` (2000 for example).
- `input_channels` - When using general_beat_saber should be `feature_size` + 4 (the number of _special states_) + output_channels (if `concat_outputs` is set, for autoregressive models. See details below)

Note: the *special states* are PAD_STATE, START_STATE, END_STATE, EMPTY_STATE, and are assigned indices in `/models/constants.py`.

#Training options

- `experiment_name` - the name to give to each run (will save the checkpoints in a folder with this name).
- `nepoch` - number of epochs to train. Note that an "epoch" is a pass through every song, but not necessarily through every part of every song, as we take only a certain number of windows (see detailed explanation below)
- `nepoch_decay` - number of epochs to train, after decaying the learning rate.
- `batch_size`
- `print_freq` - every how many iterations to print metrics
- `save_latest_freq` - every how many iterations to save the trained model
- `continue_train` - if set, it will pick up latest (or the `load_iter`) checkpoint of the model/experiment and continue training from there.
- `load_iter` - the iteration of the checkpoint to load if using `continue_train`.
- `gpu_ids` - the gpu ids to use for training (sorry, training only works with CUDA for now. Minimum setting is to set it to `0` for a single gpu)
- `workers` - number of parallel processes used to load data (can help speed up training if you have many cores / RAM)

#Detailed explanation of the training process and parameters

(for general_beat_saber dataset)

Throughout, we will use the word *point* or *time point* to refer to a position in the discrete temporal sequence for to the time-discretized versions of the song or the beat saber level. The time between one point and the next is found in the parameter `step_time`.

At each iteration of training, the _dataset_'s `__getitem__()` method is called.

`__getitem__()` gets a random song from the dataset, and it samples `num_windows` indices at random points in the song. Then for each of those points it extracts a window starting from that point and of length `input_length`.

This "mini-mini-batch" of M windows is passed as a single datapoint to the model. When you set the batch-size to N, what we do is take the M windows for each of the N calls to `__getitem__()`, and combine these two dimension to get N*M actual *items* to feed to the model. So it's as if the effective batch size is N*M. This is done in the `set_input()` method of the model (which can also do some reshaping/permuting if necessary for the model).

Each input *item* is an sequence of size `input_length`, as we mentioned. The sequence index dimension is the last, and it can have an arbitrary number of dimensions before it. For the `wavenet` model, it has a single dimension of features which contains the vector of song features at each time point, possibly concatenated with the output the model produced if using it autoregressively, which is done if the parameter `concat_outputs` is set.

`input_length` is not a parameter, but is calculated as `receptive_field + output_length 1` . `output_length` is the number of sequential outputs we want the model to predict for a single item. The `-1` is because `receptive_field` is the input_length to predict a single output (`output_length=1`). `receptive_field` should be set according to the model. For wavenet, it is computed by the model code; if the model doesn't have the attribute, `train.py` just sets it to `1`.

_Context_

There is an extra bit of complication. We can choose to pass the model some temporal context at each of the input points. This can be done with the parameter `time_shifts`. This is the size of the context window around the input point, that will be fed as a single input. For instance, if `time_shifts` is 15, then each time point is fed together with the features of 7 time points in the past and 7 time points in the future. By default this gives an extra dimension to item, unless `flatten_context` is set, which concatenates the `time_shifts` feature vectors into a long feature vector.

_Stage two dataset_

Similar, but has some differences. TODO

### Summary of time-related options

- `time_offset` - the number of points between {the last point fed as an input to the model} and {the point for which the model produces its output/prediction}
- `output_length` - the number of sequential outputs we want the model to predict for a single item of the dataset.
- `num_windows` - the number of different points in time at which a song is sampled for output of the dataset `__getitem__()`. If `num_windows=0`, this is used to mean that `__getitem__()` should return the whole sequence rather than take windows.

### Model quirks

Some of the models have special features which are dealt with in the code in a rather ad-hoc / per-model basis.

_Wavenet_

The wavenet model is autoregressive so that it reads as inputs both the song features and its own outputs. However, for predicting the output at a certain time point, it reads song features both in the past and the future of that point; while it can only read outputs for the past. This is dealt with by setting different time_offset, receptive_field, etc for the song features and the autoregressive inputs.

<!-- _Transformer_

The transformer reads the whole sequence (output from `stage_two` dataset). -->
