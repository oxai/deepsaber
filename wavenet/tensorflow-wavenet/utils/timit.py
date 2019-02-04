import numpy as np


class Timit(object):
    """Timit utilities.
    """

    # phone mapping
    # https://github.com/foundintranslation/Kaldi/blob/master/egs/timit/s4/conf/phones.60-48-39.map
    phone_map = {
        '61_to_39': {
            'aa': 'aa',
            'ae': 'ae',
            'ah': 'ah',
            'ao': 'aa',
            'aw': 'aw',
            'ax': 'ah',
            'ax-h': 'ah',
            'axr': 'er',
            'ay': 'ay',
            'b': 'b',
            'bcl': 'sil',
            'ch': 'ch',
            'd': 'd',
            'dcl': 'sil',
            'dh': 'dh',
            'dx': 'dx',
            'eh': 'eh',
            'el': 'l',
            'em': 'm',
            'en': 'n',
            'eng': 'ng',
            'epi': 'sil',
            'er': 'er',
            'ey': 'ey',
            'f': 'f',
            'g': 'g',
            'gcl': 'sil',
            'h#': 'sil',
            'hh': 'hh',
            'hv': 'hh',
            'ih': 'ih',
            'ix': 'ih',
            'iy': 'iy',
            'jh': 'jh',
            'k': 'k',
            'kcl': 'sil',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'ng': 'ng',
            'nx': 'n',
            'ow': 'ow',
            'oy': 'oy',
            'p': 'p',
            'pau': 'sil',
            'pcl': 'sil',
            'q': '',
            'r': 'r',
            's': 's',
            'sh': 'sh',
            't': 't',
            'tcl': 'sil',
            'th': 'th',
            'uh': 'uh',
            'uw': 'uw',
            'ux': 'uw',
            'v': 'v',
            'w': 'w',
            'y': 'y',
            'z': 'z',
            'zh': 'sh',
        }
    }

    _61_phones = sorted(list(phone_map['61_to_39'].keys()))
    _39_phones = sorted(set(phone_map['61_to_39'].values()))
    _39_phones.remove('')

    @staticmethod
    def convert_61_to_39(origin_phones):
        """Convert 61 phones set to 39 one.
        """

        ret = []
        for phn in origin_phones:
            map_phn = None
            if isinstance(phn, str):
                map_phn = Timit.phone_map['61_to_39'][phn]
            elif isinstance(phn, int) or issubclass(type(phn), np.integer):
                map_phn = Timit.phone_map['61_to_39'][Timit._61_phones[phn]]
            else:
                raise TypeError('Type of each phone should be either str of int.')

            # 'q' will disappear in 39 phones set
            if map_phn in Timit._39_phones:
                ret.append(Timit._39_phones.index(map_phn))

        return ret

