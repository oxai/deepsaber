from pathlib import Path
import numpy as np
import librosa
from base.data.base_dataset import BaseDataset
import json


class SongDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError(f'Invalid directory: {opt.data_dir}')
        self.audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        self.level_jsons = sorted(data_path.glob(f'**/{self.opt.level_diff}.json'), key=lambda path: path.parent.__str__())
        assert self.audio_files, "List of audio files cannot be empty"
        assert self.level_jsons, "List of level files cannot be empty"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=22050, type=float)
        parser.add_argument('--level_diff', default='Easy', help='Difficulty level for beatsaber level')
        return parser

    def name(self):
        return "SongDataset"

    def __getitem__(self, item):
        y, sr = librosa.load(self.audio_files[item], sr=self.opt.sampling_rate)
        level = json.load(open(self.level_jsons[item], 'r'))
        # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        hop_length = 512
        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Beat track on the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
        #print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        # Convert the frame indices of beat events into timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        # And the first-order differences (delta features)
        mfcc_delta = librosa.feature.delta(mfcc)
        # Stack and synchronize between beat events
        # This time, we'll use the mean value (default) instead of median
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
        # Selects peak positions on the onset strength curve
        onset = librosa.onset.onset_detect(y=y, sr=sr)
        # Finally, stack all beat-synchronous features together
        beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
        return {'input': y, 'target': level, 'features': beat_features}

    def __len__(self):
        return len(self.audio_files)
