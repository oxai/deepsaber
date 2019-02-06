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