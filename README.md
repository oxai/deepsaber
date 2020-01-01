Google Doc: https://docs.google.com/document/d/1UDSphLiWsrbdr4jliFq8kzrJlUVKpF2asaL65GnnfoM/edit

Join our discord here! https://discord.gg/T6djf8N

Welcome to the readme for DeepSaber, an automatic generator of BeatSaber levels. There is a lot of stuff here, fruit of a lot of work by the team in [OxAI Labs](http://oxai.org/labs). Contact me at guillermo . valle at oxai.org , or on twitter (@guillefix) for any questions/suggestions!

# TLDR generation

_Requirements/Dependencies_

From Pypi, using pip:
- numpy
- librosa
- pytorch (installed as `torch` or via https://pytorch.org/get-started/locally/)
- pandas
- matplotlib
- pillow

From your favorite package manager:
- [sox](http://sox.sourceforge.net/) (e.g. `sudo apt-get install sox`)
- ffmpeg

Reccommended hardware:
- Nvidia GPU with CUDA [:/ unfortunately, stage 2 is too slow in CPU (although it should work in theory.., after removing "cuda" options in "./scrit_generate.sh" below]



(Do this first time generating) Download pre-trained weights from https://mega.nz/#!tJBxTC5C!nXspSCKfJ6PYJjdKkFVzIviYEhr0BSg8zXINBqC5rpA, and extract the contents (two folders with four files in total) inside the folder `scripts/training/`.

Then, *to generate a level simply run* (if on linux):

`cd scripts/generation`

`./script_generate.sh [path to song]`

where you should substitute `[path to song]` with the path to the song which you want to use to generate the level, which should be on *wav* format (sorry). Also it doesn't like spaces in the filename :P . Generation should take about 3 minutes for a 3 minutes song, but it grows (I think squared-ly) with the length, and it will depend on how good your GPU is (mine is a gtx 1070).

This will generate a zip with the Beat Saber level which should be found in `scripts/generation/generated`. You should be able to put it in the custom levels folders in the current version of DeepSaber (as of end of 2019).

I also recommending reading about how to use the "open_in_browser" option, described in the next section, which is quite a nice feature to visualize the generated quickly and easy to set up if you have dropbox.

*Pro tip*: If the generated level doesn't look good (this is deep learning, it's hard to give guarantees :P), try changing in `./script_generate.sh`

```sh
cpt2=2150000
#cpt2=1200000
#cpt2=1450000
```
to
```sh
#cpt2=2150000
#cpt2=1200000
cpt2=1450000
```
See below for explanation


# Further generation options

[TODO] make this more user friendly.

If you open the script `scripts/generation/script_generate.sh` in your editor, you can see other options. You can change `exp1` and `exp2`, as well as the corresponding `cpt1` and `cpt2`. These correspond to "experiments" and "checkpoints", and determine where to get the pre-trained network weights. The checkpoints are found in folders inside `scripts/training`, and `cpt1`/`cpt2`, just specify which of the saved iterations to use. If you train your own models, you can change those to generate using your trained models. You can also change them to explore different pre-trained versions available at https://mega.nz/#!VEo3XAxb!7juvH_R_6IjG1Iv_sVn1yGFqFY3sQVuFyvlbbdDPyk4 (for example DeepSaber 1 used the latest in "block_placement_new_nohumreg" for stage 1 and the latest in "block_selection_new"), but the one you downloaded above is the latest one (DeepSaber 2, trained on a more curated dataset), so should typically work best (but there is always some stochasticity and subjectivity so).

You can also change the variable `type` from `deepsaber` to `ddc` to use [DDC](https://github.com/chrisdonahue/ddc) as the stage 1 (where in times to put notes), while still using deepsaber for stage 2 (which notes to put at each instant for which stage 1 decides to put something). But this requires setting up DDC first. If you do, then just pass the generated stepmania file as a third command argument, and it should work the same.

There is also an "open in browser" option (which is activated by uncommenting the line `#--open_in_browser` inside the `deepsaber` if block), which is very useful for testing, as it gives you a link with a level visualizer on the broser. To set it up, you just need to set up the script `scripts/generation/dropbox_uploader.sh`. This is very easy, just run the script, and it will guide you with how to link it to your dropbox account (you need one.).

A useful parameter to change also is the `--peak threshold`. It is currently set at about `0.33`, but you can experiment with it. Putting it higher, makes it output less notes, and putting it lower, makes more notes.

If you dig deeper, you can also disable the option `--use_beam_search`, but the outputs are then usually quite random -- you can also try setting the `--temperature` parameter lower to make it a less so, but beam search is typically better.

Digging even deeper, there is a very hidden option :P inside `scripts/generation/generate_stage2.py` in line 59, there `opt["beam_size"] = 17`. You can change this number if you want. Making it larger means the generation will take longer but it will typically be of higher quality (it's as if the model thinks harder about it), and making it smaller has the opposite effect, but can be a good thing to try if you want fast generation for some reason.

You could change `opt["n_best"] = 1` to something greater than 1, and change some other code, to get outputs that model thought "less likely" and explore what the model can generate [contact me for more details].

# Example of whole pipeline

_Requirements/Dependencies_
- numpy
- librosa
- pytorch
- mpi4py (only for training->data_processing)

This is a quick run through the whole pipeline, from getting data, to training to generating:

Run all this in root folder of repo

_Get example data_

`wget -O DataSample.tar.gz https://www.dropbox.com/s/2i75ebqmm5yd15c/DataSample.tar.gz?dl=1`

[Can also download the whole dataset here: https://mega.nz/#!sABVnYYJ!ZWImW0OSCD_w8Huazxs3Vr0p_2jCqmR44IB9DCKWxac]

`tar xzvf DataSample.tar.gz`

`mv scripts/misc/bash_scripts/extract_zips.sh DataSample/`

`cd DataSample; ./extract_zips.sh`

`rm DataSample/*.zip`

`mv DataSample/* data/extracted_data`

_Get reduced state list_

`wget -O data/statespace/sorted_states.pkl https://www.dropbox.com/s/ygffzawbipvady8/sorted_states.pkl?dl=1`

_Data augmentation (optional)_

`scripts/data_processing/augment_data.sh`

_extract features_

Dependencies: librosa, mpi4py (and mpi itself). TODO: make mpi an optional dependency.

You can change the "`Expert,ExpertPlus`" with any comma-separated (and with no spaces) list of difficulties to train on levels of those difficulties.

`mpiexec -n $(nproc) python3 scripts/feature_extraction/process_songs.py data/extracted_data Expert,ExpertPlus --feature_name multi_mel --feature_size 80`

`mpiexec -n $(nproc) python3 scripts/feature_extraction/process_songs.py data/extracted_data Expert,ExpertPlus --feature_name mel --feature_size 100`

_pregenerate level tensors (new fix that makes stage 1 training much faster)_

The way this works is that we need to run this command for each difficulty level we want to train on. Here Expert and ExpertPlus

`mpiexec -n 12 python3 scripts/feature_extraction/process_songs_tensors.py ../../data/DataSample/ Expert --replace_existing --feature_name multi_mel --feature_size 80`

`mpiexec -n 12 python3 scripts/feature_extraction/process_songs_tensors.py ../../data/DataSample/ ExpertPlus --replace_existing --feature_name multi_mel --feature_size 80`

_training_

Dependencies: pytorch

Train Stage 1. Either of two options:
 * (wavenet_option): `scripts/training/debug_script_block_placement.sh`
 * (ddc option): `scripts/training/debug_script_ddc_block_placement.sh`

Train Stage 2: `scripts/training/debug_script_block_selection.sh`

_generation (using the model trained as above)_

To generate with the models trained as above, you need to edit `scripts/generation/script_generate.sh` and change the variable `exp1` to the experiment name from which we want to get the trained weights: if following the example above it would be either `test_block_placement` or `test_ddc_block_placement` if used ddc; change the variable `exp2` to `test_block_selection`, change `cpt1` to the latest block placement iteration, and `cpt2` to the latest block selection iteration. The latest iterations can be found by looking for files in the folders in `scripts/training/` with the names of the different experiments have the form `iter_[checkpoint]_net_.pth`.

<!-- The last argument is the path to a song in wav format

`scripts/generation/script_generate.sh deepsaber [checkpoint1] [checkpoint2] [path to some song in wav format]` -->

To use the ddc options, or the "open in browser" option requires more setting up (specially the former). But the above should generate a zip file with the level.

* The "open in browser" option is very useful for visualizing the level. You just need to set up the script `scripts/generation/dropbox_uploader.sh`. This is very easy, just run the script, and it will guide you with how to link it to your dropbox account (you need one.)

* The DDC option requires setting up DDC (https://github.com/chrisdonahue/ddc), which now includes a docker component, and requires its own series of steps. But hopefully the new trained model will supersede this.

# Getting the data

[TODO] Here we describe the scripts to scrap Beastsaver and BeastSaber to get the training data

### _download data_

`scripts/data_retrieval/download_data.py`

### _obtain the most common states to use for the reduced state representation_

`scripts/data_processing/state_space_functions.py`

# train

## _prepare and preprocess data_

_data augmentation_

`scripts/data_processing/augment_data.sh`

_data preprocessing_

`scripts/feature_extration/process_songs.py` []

##_training_

`scripts/training/script_block_placement.sh`

See more at readme in `scripts/training/README.md`


<!-- ## Minimal Example of Usage
```python
from base.options.train_options import TrainOptions
from base.data import create_dataset, create_dataloader
from base.models import create_model

opt = TrainOptions().parse()
dataset = create_dataset(opt)
dataloader = create_dataloader(dataset)
data = dataloader[0]
#{'input': songs, 'target': notes, 'features': beat_features}
model = create_model(opt)

for epoch in range(opt.epoch_count, opt.nepoch + opt.nepoch_decay):
    for i, data in enumerate(dataloader)
        model.set_input(data)
        model.optimize_parameters()
        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            print(losses)
    print(f'End of epoch {i}')
    model.update_learning_rate()
``` -->

<!-- ### Dataset
Steps for creating a custom dataset with custom command line options:
*  Create a file 'datasetname_dataset.py' in the 'data' folder, where datasetname is your custom name.
*  Import the class BaseDataset from base.data.base_dataset
*  Create a class DatasetNameDataset (case does not matter) that inherits from BaseDataset
*  Define the \_\_getitem\_\_ and \_\_len\_\_ methods
*  Options are stored in the opt object that is passed to \_\_init\_\_.
*  Create the 'modify commandline options method' to add your custom commandline options - e.g.:
```python
@staticmethod
def modify_commandline_options(parser, is_train):
    parser.add_argument('--sampling_rate', default=22050, type=float)
    parser.add_argument('--level_diff', default='Expert', help='Difficulty level for beatsaber level')
    parser.add_argument('--hop_length', default=512, type=int)  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    parser.add_argument('--compute_feats', action='store_true', help="Whether to extract musical features from the song")
    return parser
```
*  In this example, sampling\_rate, level\_diff, hop\_length and compute\_feats are accessible as attributes of the option object.
*  Add a name() method that returns the name of your class as a string, e.g.
```python
    def name(self):
        return "SongDataset"
```
*  When launching your training script, add the command-line argument --dataset_name=yourDatasetName
*  !!! See base.data.song_dataset.py for an example of a dataset that follows this API !!!

### Model
Steps for creating a custom model with custom command line options:
*  Create a file 'modelname_model.py' in the 'models' folder, where modelname is your custom name.
*  Import the class BaseModel from base.data.base_model
*  Create a class ModelNameModel (case does not matter) that inherits from BaseDataset
*  Initialize a nn.Module instance (your pytorch neural network) with a forward() method, and add it to the ModelNameModel instance as an attribute with name modelname_net
*  Append the module name to the attribute list: self.module\_names.append('modelname') as a string in \_\_init\_\_
*  Add a forward method where the loss is computed. The name of the loss must be self.loss\_lossname.
*  Append lossname to the self.loss\_names list as a string in \_\_init\_\_
*  Add a backward() method where the optimizers are set up, gradients are computed on the losses using self.loss\_lossname.backward(), and an optimizer step is performed.
*  Optimizers cna be stored in the list (self.optimizers)
*  Add an optimize\_parameters() method where self.set\_requires\_grad(self.modelname\_net, requires\_grad=True), self.forward() and self.backward() are defined.
*  Example:
```python
    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['crossentropy']
        self.metric_names = []
        self.module_names = ['wave']
        self.image_paths = []
        self.schedulers = []
        self.net = WaveNet(layers=opt.layers,
                           blocks=opt.blocks,
                           dilation_channels=opt.dilation_channels,
                           residual_channels=opt.residual_channels,
                           skip_channels=opt.skip_channels,
                           end_channels=opt.end_channels,
                           input_channels=opt.input_channels,
                           output_length=opt.output_length,
                           kernel_size=opt.kernel_size,
                           bias=opt.bias)
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]

        @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--layers', type=int, default=10, help="Number of layers in each block")
        parser.add_argument('--blocks', type=int, default=4, help="Number of residual blocks in network")
        parser.add_argument('--dilation_channels', type=int, default=32, help="Number of channels in dilated convolutions")
        parser.add_argument('--residual_channels', type=int, default=32, help="Number of channels in the residual link")
        parser.add_argument('--skip_channels', type=int, default=256)
        parser.add_argument('--end_channels', type=int, default=256)
        parser.add_argument('--input_channels', type=int, default=1)
        parser.add_argument('--output_length', type=int, default=1)
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--bias', action='store_false')
        return parser

    def forward(self):
        self.output = self.wave_net.forward(self.input)
        self.loss_crossentropy = F.cross_entropy(self.output, self.target)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_crossentropy.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()
        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass
```
*  Create the 'modify commandline options method' to add your custom commandline options - (e.g. look above)
*  Add a name() method that returns the name of your class as a string, e.g.
```python
    def name(self):
        return "WaveNetModel"
```
*  !!! See base.models.wavenet_model.py for an example of a dataset that follows this API !!!

# Notes
1.  If the output of your dataset is a dictionary: data = {'input': input\_tensor, 'target': target\_tensor}, you can use model.set_input(data) to store input and target into your model for use in forward.
2.  Store the nn.Module instance in another file (e.g. networks.py) for better abstraction. -->
