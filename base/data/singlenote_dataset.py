from pathlib import Path
from itertools import tee
import numpy as np
import torch
import librosa
from base.data.base_dataset import BaseDataset
import json


class SinglenoteDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError(f'Invalid directory: {opt.data_dir}')
        self.audio_files = sorted(data_path.glob('**/*.ogg'), key=lambda path: path.parent.__str__())
        self.level_jsons = []
        audio_no_level = []  # store audios for which there is no json level
        for i, path in enumerate(self.audio_files):
            try:
                level = list(path.parent.glob(f'./{self.opt.level_diff}.json'))[0]
                self.level_jsons.append(level)
            except IndexError:
                audio_no_level.append(i)
        for i in reversed(audio_no_level):  # not to throw off preceding indices
            self.audio_files.pop(i)
        assert self.audio_files, "List of audio files cannot be empty"
        assert self.level_jsons, "List of level files cannot be empty"
        assert len(self.audio_files) == len(self.level_jsons)
        self.eps = 0.01

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=11025, type=float)
        parser.add_argument('--level_diff', default='Expert', help='Difficulty level for beatsaber level')
        parser.add_argument('--hop_length', default=256, type=int)  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        parser.add_argument('--compute_feats', action='store_true', help="Whether to extract musical features from the song")
        parser.add_argument('--padded_length', type=int, default=3000000)
        parser.add_argument('--chunk_length', type=int, default=9000)
        return parser

    def name(self):
        return "SongDataset"

    def __getitem__(self, item):
        y, sr = librosa.load(self.audio_files[item], sr=self.opt.sampling_rate)
        y = librosa.util.fix_length(y, size=self.opt.padded_length)
        level = json.load(open(self.level_jsons[item], 'r'))
        y_harmonic, y_percussive = librosa.effects.hpss(y)  # Separate harmonics and percussive into two waveforms
        tempo, beat_times = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=self.opt.hop_length,
                                                    bpm=level['_beatsPerMinute'], units='time')
        time_between_beats = [beat_t1 - beat_t0 for beat_t0, beat_t1 in pairwise([0] + list(beat_times))]
        beat_time_to_note = {note['_time']: [note['_cutDirection'], note['_lineIndex'],
                            note['_lineLayer'], note['_type']] for note in level['_notes']}
        time = np.array([round(i / sr, 2) for i in range(len(y))])
        target = []
        yy = []
        for note_beat_time in beat_time_to_note.keys():
            beat_idx = round(note_beat_time)  # get integer beat index
            if beat_idx < len(beat_times.tolist()):
                note_time = (note_beat_time - beat_idx) * time_between_beats[beat_idx] + beat_times[beat_idx]  # time position of note
                note_time_idx = np.where(np.absolute(time - note_time) < self.eps)[0][0]  # chunk end point
                start_idx = np.where(np.absolute(note_time - time_between_beats[beat_idx] - time) < self.eps)[0][0]  # chunk starting point
                pad_extent = self.opt.chunk_length - (note_time_idx - start_idx)
                padded_chunk_before_beat = np.pad(y[start_idx: note_time_idx], (pad_extent, 0), mode='constant')
                yy.append(padded_chunk_before_beat)  # only pad beginning of sequence
                target.append(beat_time_to_note[note_beat_time])  # write note at time position of beat
        target = torch.tensor(target).unsqueeze(0)  # transform into (2+1)D tensor
        target += 1  # so that class range is > 0
        yy = torch.tensor(yy).unsqueeze(0)  # adding channel dim
        yy = (yy - yy.mean())/torch.abs(yy).max()
        return {'input': yy, 'target': target}

    def __len__(self):
        return len(self.audio_files)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
