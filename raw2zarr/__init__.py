"""
raw2zarr
========

High-level API for converting raw radar data to Zarr.

Author: Alfonso Ladino
Email: alfonso8@illinois.edu
Version: 2025.1.0
"""

__author__ = "Alfonso Ladino"
__email__ = "alfonso8@illinois.edu"
__version__ = "2025.1.0"

from raw2zarr.builder.dtree_radar import radar_datatree
from raw2zarr.builder.runner import (
    append_parallel,
    append_sequential,
)
from raw2zarr.writer.zarr_writer import dtree2zarr

__all__ = [
    "radar_datatree",
    "append_sequential",
    "append_parallel",
    "dtree2zarr",
]
