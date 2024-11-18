"""
raw2zarr
======

Top-level package for raw2zarr.

"""

__author__ = """Alfonso Ladino"""
__email__ = "alfonso8@illinois.edu"

from .dtree_builder import datatree_builder, append_sequential, append_parallel
from .data_reader import accessor_wrapper
from .utils import ensure_dimension, fix_angle, batch, dtree_encoding

__all__ = [
    "datatree_builder",
    "append_sequential",
    "append_parallel",
    "accessor_wrapper",
    "ensure_dimension",
    "fix_angle",
    "batch",
    "dtree_encoding",
]
