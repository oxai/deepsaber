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
*  When launching your training script, add the command-line argument --dataset_name=yourDatasetName