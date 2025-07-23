from xarray import DataTree
from xradar.io import open_nexradlevel2_datatree

from ..preprocess import normalize_input_for_xradar


def nexradlevel2_loader(file) -> DataTree:
    """
    Loads NEXRAD Level 2 files from local or S3, decompresses if needed.
    """
    path = normalize_input_for_xradar(file)
    return open_nexradlevel2_datatree(path)
