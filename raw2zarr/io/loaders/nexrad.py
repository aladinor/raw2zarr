from xarray import DataTree
from xradar.io import open_nexradlevel2_datatree

from ..preprocess import normalize_input_for_xradar


def nexradlevel2_loader(file) -> DataTree:
    """
    Loads NEXRAD Level 2 files from local or S3, decompresses if needed.
    Returns a local uncompressed path to xradar, which does not support streaming.
    """
    # TODO: use prepare2read when xradar supports streaming for nexradlevel2 engine.
    #  See https://github.com/openradar/xradar/issues/265
    path = normalize_input_for_xradar(file)
    return open_nexradlevel2_datatree(path)
