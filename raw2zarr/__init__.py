"""
raw2zarr
========

High-level API for converting raw radar data to Zarr.

Author: Alfonso Ladino
Email: alfonso8@illinois.edu
Version: 0.5.0
"""

__author__ = "Alfonso Ladino"
__email__ = "alfonso8@illinois.edu"
__version__ = "0.5.0"

from raw2zarr.builder.dtree_radar import radar_datatree
from raw2zarr.builder.executor import (
    append_parallel,
    append_sequential,
)
from raw2zarr.writer.zarr_writer import dtree_to_zarr

__all__ = [
    "radar_datatree",
    "append_sequential",
    "append_parallel",
    "dtree_to_zarr",
]
