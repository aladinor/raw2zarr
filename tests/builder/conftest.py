import numpy as np
import pytest
from packaging.version import parse as parse_version

import zarr

_numpy_version = parse_version(np.__version__)
_zarr_version = (
    parse_version(zarr.__version__)
    if hasattr(zarr, "__version__")
    else parse_version("0")
)

requires_numpy2 = pytest.mark.skipif(
    _numpy_version < parse_version("2.0.0"),
    reason="Requires NumPy >= 2.0.0 due to StringDType encoding behavior.",
)

requires_zarr3 = pytest.mark.skipif(
    _zarr_version < parse_version("3.0.0"),
    reason="Requires Zarr >= 3.0.0 for format=3 compatibility.",
)

requires_numpy2_and_zarr3 = pytest.mark.skipif(
    (_numpy_version < parse_version("2.0.0"))
    or (_zarr_version < parse_version("3.0.0")),
    reason="Requires NumPy >= 2.0.0 and Zarr >= 3.0.0",
)
