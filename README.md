Google Doc: https://docs.google.com/document/d/1UDSphLiWsrbdr4jliFq8kzrJlUVKpF2asaL65GnnfoM/edit

## Quickest testing procedure:

Run all this in root folder of repo

_Get data_

`wget -O DataSample.tar.gz https://www.dropbox.com/s/2i75ebqmm5yd15c/DataSample.tar.gz?dl=1`

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

`scripts/feature_extraction/python3 process_songs.py data/extracted_data Expert,ExpertPlus --feature_name multi_mel --feature_size 80`

`scripts/feature_extraction/python3 process_songs.py data/extracted_data Expert,ExpertPlus --feature_name mel --feature_size 100`

_training_

Train Stage 1. Either of two options:
 * (wavenet_option): `scripts/training/debug_script_block_placement.sh`
 * (ddc option): `scripts/training/debug_script_ddc_block_placement.sh`

Train Stage 2: `scripts/training/debug_script_block_selection.sh`

_generation_

In the command below, after training substitute "checkpoint1" with the latest iteration number which appears saved in the folder `scripts/training/test_block_placement` or `scripts/training/test_ddc_block_placement` if used ddc; substitute "checkpoint2" with the latest iteration number which appears saved in the folder `scripts/training/test_block_selection`. The files in those folders have the form `iter_[checkpoint]_net_.pth`

The last argument is the path to a song in wav format

`scripts/generation/script_generate.sh deepsaber [checkpoint1] [checkpoint2] [path to some song in wav format]`

To use the ddc options, or the "open in browser" option requires more setting up (specially the former). But the above should generate a zip file with the level.

* The "open in browser" option is very useful for visualizing the level. You just need to set up the script `scripts/generation/dropbox_uploader.sh`. This is very easy, just run the script, and it will guide you with how to link it to your dropbox account (you need one.)

* The DDC option requires setting up DDC (https://github.com/chrisdonahue/ddc), which now includes a docker component, and requires its own series of steps. But hopefully the new trained model will supersede this.

# Getting the data

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

`scripts/training/train.py`

`scripts/training/script_block_placement.sh`

# generate

---------------

# Beatsaber level generator @OxAI

## Requirements
- numpy
- librosa
- pytorch

## Minimal Example of Usage
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
```

### Dataset
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
2.  Store the nn.Module instance in another file (e.g. networks.py) for better abstraction.
