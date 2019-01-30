import sys
import pytest
import base.options as options
from base.utils import utils

"""
Configuration file used by pytest before running tests.
Importing fixtures here allows usage in test files.
https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
"""


@pytest.fixture
def wsi_file():
    '''Returns an example WSI file path'''
    if utils.on_cluster():
        return '/well/rittscher/projects/TCGA_prostate/TCGA/ff46403a-5498-4ffa-bf85-73afb558eb95/TCGA-J4-A67R-01Z-00-DX1.833DA729-D98E-44F8-A437-1B5BF52071BD.svs'
    elif sys.platform == 'linux':
        return '/home/sedm5660/Documents/Temp/Data/cancer_phenotype/17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi'
    else:
        return '/Users/andreachatrian/Documents/Temp/Data/cancer_phenotype/17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi'


@pytest.fixture
def tcga_data():
    if utils.on_cluster():
        return None  # TODO look path up
    elif sys.platform == 'linux':
        sample_path = '/home/sedm5660/Documents/Temp/Data/cancer_phenotype/tcga_data_info/prad_tcga_pan_can_atlas_2018/data_clinical_sample.txt'
        cna_path = '/home/sedm5660/Documents/Temp/Data/cancer_phenotype/tcga_data_info/prad_tcga_pan_can_atlas_2018/data_CNA.txt'
    else:
        sample_path = '/Users/andreachatrian/Documents/Temp/Data/cancer_phenotype/tcga_data_info/prad_tcga_pan_can_atlas_2018/data_clinical_sample.txt'
        cna_path = '/Users/andreachatrian/Documents/Temp/Data/cancer_phenotype/tcga_data_info/prad_tcga_pan_can_atlas_2018/data_CNA.txt'
    return {
        'sample': sample_path,
        'cna': cna_path
    }



@pytest.fixture
def train_options():
    TrainOptions = options.train_options.TrainOptions
    train_options = TrainOptions()
    sys.argv = sys.argv[1:]  # must remove file name, which is first argument
    train_options.parser.set_defaults(gpu_ids='-1')
    return train_options


@pytest.fixture
def test_options():
    TestOptions = options.test_options.TestOptions
    test_options = TestOptions()
    sys.argv = sys.argv[1:]  # must remove file name, which is first argument
    test_options.parser.set_defaults(gpu_ids='-1')
    return test_options


@pytest.fixture
def apply_options():
    ApplyOptions = options.apply_options.ApplyOptions
    apply_options = ApplyOptions()
    apply_options.parser.set_defaults(gpu_ids='-1')
    sys.argv = sys.argv[1:]  # must remove file name, which is first argument
    return apply_options
