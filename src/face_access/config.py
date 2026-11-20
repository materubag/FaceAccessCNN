import copy
from ._profile import PROFILE
from .common import canonical, read_json

PREPROCESSING = dict(size=96, channels=1, stored_channels=3, grayscale=True,
                     filter="median", kernel=3, normalization="uint8/255", range=[0, 1], align_eyes=False)


def detector_profile():
    return copy.deepcopy(PROFILE)


def input_contract():
    return dict(detector=detector_profile(), preprocessing=copy.deepcopy(PREPROCESSING))


def contract_hash():
    return canonical(input_contract())


def load_config(path):
    """Los .yaml son JSON válido (subconjunto de YAML1.2); no se necesita PyYAML."""
    return read_json(path)
