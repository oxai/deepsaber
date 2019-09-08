import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.append(MODELS_DIR)

import transformer.constants
import transformer.Modules
import transformer.Layers
import transformer.SubLayers
import transformer.Models
import transformer.Translator
import transformer.Beam
import transformer.Optim

__all__ = [
    transformer.constants, transformer.Modules, transformer.Layers,
    transformer.SubLayers, transformer.Models, transformer.Optim,
    transformer.Translator, transformer.Beam]
