Google Doc: https://docs.google.com/document/d/1UDSphLiWsrbdr4jliFq8kzrJlUVKpF2asaL65GnnfoM/edit


# Beatsaber level generator @OxAI

## Requirements
- numpy
- librosa
- pytorch

## Example usage
```python
from base.options.train_options import TrainOptions
from base.data import create_dataset, create_dataloader

opt = TrainOptions().parse()
dataset = create_dataset(opt)
dataloader = create_dataloader(dataset)
data = dataloader[0]
#{'input': songs, 'target': levels, 'features': beat_features}
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
*  Add an optimize\_parameters() method where self.set\_requires\_grad(self.modelname\_net, requires\_grad=True), self.forward() and self.backward() are defined
*  Create the 'modify commandline options method' to add your custom commandline options - e.g.:
```python
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
```
*  Add a name() method that returns the name of your class as a string, e.g.
```python
    def name(self):
        return "WaveNetModel"
```
