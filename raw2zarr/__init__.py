"""
raw2zarr
======

Top-level package for raw2zarr.

"""

__author__ = """Alfonso Ladino"""
__email__ = "alfonso8@illinois.edu"
__version__ = "2025.1.0."

from .dtree_builder import (
    datatree_builder,
    append_sequential,
    append_parallel,
    process_file,
)
from .dtree_io import prepare2read, load_radar_data, _load_file
from .utils import ensure_dimension, fix_angle, batch, dtree_encoding
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
