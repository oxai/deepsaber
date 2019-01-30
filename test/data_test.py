import sys
from base.data import create_dataset


def test_input(train_options):
    sys.argv.append('-d=/Users/andreachatrian/Documents/Repositories/oxai/beatsaber/DataE')
    opt = train_options.parse()
    song_dataset = create_dataset(opt)
    data = song_dataset[0]
