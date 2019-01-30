import sys
import pytest
from base.options.train_options import TrainOptions
from base.options.test_options import TestOptions

"""
Configuration file used by pytest before running tests.
Importing fixtures here allows usage in test files.
https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
"""




@pytest.fixture
def train_options():
    train_options = TrainOptions()
    sys.argv = sys.argv[1:]  # must remove file name, which is first argument
    train_options.parser.set_defaults(gpu_ids='-1')
    return train_options


@pytest.fixture
def test_options():
    test_options = TestOptions()
    sys.argv = sys.argv[1:]  # must remove file name, which is first argument
    test_options.parser.set_defaults(gpu_ids='-1')
    return test_options