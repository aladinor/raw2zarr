"""
raw2zarr
======

Top-level package for raw2zarr.

"""

__author__ = """Alfonso Ladino"""
__email__ = "alfonso8@illinois.edu"
__version__ = "2025.1.0."

from .dtree_builder import (
    append_parallel,
    append_sequential,
    datatree_builder,
    process_file,
)
from .dtree_io import _load_file, load_radar_data, prepare2read
from .utils import batch, dtree_encoding, ensure_dimension, fix_angle
from .zarr_writer import dtree2zarr

__all__ = [
    "datatree_builder",
    "append_sequential",
    "append_parallel",
    "load_radar_data",
    "ensure_dimension",
    "fix_angle",
    "batch",
    "dtree_encoding",
    "process_file",
    "prepare2read",
    "_load_file",
    "dtree2zarr",
]
