import gzip
import tempfile

import fsspec
import pytest
from xarray import DataTree


@pytest.fixture(scope="session")
def nexrad_aws_file_sweep10_misaligned(tmp_path_factory):
    """
    Download a real NEXRAD file with known azimuth and angle misalignment (sweep_10) from AWS.
    Used for fix_angle/fix_azimuth tests.
    """
    aws_url = "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_154426_V06"
    local_path = tmp_path_factory.mktemp("nexrad_data") / "KILX20230629_154426_V06"
    with fsspec.open(aws_url, anon=True) as s3_file:
        with open(local_path, "wb") as local_file:
            local_file.write(s3_file.read())
    return str(local_path)


@pytest.fixture(scope="session")
def nexrad_local_uncompressed_file(tmp_path_factory):
    """
    Download a NEXRAD radar file from AWS S3 to a temporary directory for testing.
    """
    aws_url = "s3://noaa-nexrad-level2/2012/01/29/KVNX/KVNX20120129_000840_V06.gz"
    local_path = tmp_path_factory.mktemp("nexrad_data") / "KVNX20120129_000840_V06.gz"
    with fsspec.open(aws_url, anon=True) as s3_file:
        with open(local_path, "wb") as local_file:
            local_file.write(s3_file.read())
    with (
        gzip.open(str(local_path), "rb") as gz,
        tempfile.NamedTemporaryFile(delete=False) as temp_file,
    ):
        temp_file.write(gz.read())
        temp_file_path = temp_file.name
    return temp_file_path


@pytest.fixture(scope="session")
def nexrad_local_gz_file(tmp_path_factory):
    """
    Download a GZIP NEXRAD file from AWS S3 for testing.
    """
    aws_url = "s3://noaa-nexrad-level2/2012/01/29/KVNX/KVNX20120129_000840_V06.gz"
    gz_path = tmp_path_factory.mktemp("nexrad_data") / "KILX20230629_154426_V06.gz"
    with fsspec.open(aws_url, anon=True) as s3_file:
        with open(gz_path, "wb") as local_file:
            local_file.write(s3_file.read())
    return str(gz_path)


@pytest.fixture(scope="session")
def nexrad_s3_file(tmp_path_factory):
    """
    Passing s3 file directly
    """
    return "s3://noaa-nexrad-level2/2012/01/29/KVNX/KVNX20120129_000840_V06.gz"


@pytest.fixture(scope="session")
def nexrad_aws_file_SAILS(tmp_path_factory):
    """
    Download a NEXRAD radar file from AWS S3 to a temporary directory for testing.
    """
    aws_url = "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06"
    local_path = tmp_path_factory.mktemp("nexrad_data") / "KILX20230629_161526_V06"
    with fsspec.open(aws_url, anon=True) as s3_file:
        with open(local_path, "wb") as local_file:
            local_file.write(s3_file.read())
    return str(local_path)


@pytest.fixture(scope="session")
def iris_aws_file():
    """
    Download an IRIS radar file from AWS S3 to a temporary directory for testing.
    """
    return "s3://s3-radaresideam/l2_data/2022/06/01/Guaviare/GUA220601112817.RAWP3AL"


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    """
    Parameterize whether to use a file path or a file-like object for testing.
    """
    return request.param


@pytest.fixture(scope="session")
def nexradlevel2_files(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def radar_dtree_factory():
    """
    Factory fixture that returns a function to create a radar-style DataTree
    with all variables aligned along the specified append_dim (default: 'vcp_time').
    """

    def _create_radar_dtree(append_dim: str = "vcp_time") -> DataTree:
        import numpy as np
        import pandas as pd
        import xarray as xr

        vcp_time = pd.Timestamp("2011-05-20T00:00:23Z")

        # Root dataset
        root_data = xr.Dataset(
            coords={append_dim: [vcp_time]},
            data_vars={
                "volume_number": (append_dim, [0]),
                "platform_type": (append_dim, ["fixed"]),
                "instrument_type": (append_dim, ["radar"]),
                "time_coverage_start": (append_dim, ["2011-05-20T00:00:23Z"]),
                "time_coverage_end": (append_dim, ["2011-05-20T00:04:31Z"]),
                "longitude": (append_dim, [-98.13]),
                "latitude": (append_dim, [36.74]),
                "altitude": (append_dim, [378]),
            },
        )

        # Axes
        azimuth = np.round(np.linspace(0.25, 359.75, 720), 4)
        range_ = np.round(np.linspace(2125, 459900, 1832), 1)
        az_len, rg_len = len(azimuth), len(range_)

        # Georeferenced 2D fields
        x = np.random.uniform(-2000, 2000, size=(az_len, rg_len))
        y = np.random.uniform(2000, 460000, size=(az_len, rg_len))
        z = np.random.uniform(400, 17000, size=(az_len, rg_len))

        # Sweep dataset
        sweep_data = xr.Dataset(
            coords={
                append_dim: [vcp_time],
                "azimuth": azimuth,
                "range": range_,
                "elevation": ("azimuth", np.full(az_len, 0.5273)),
                "time": (
                    "azimuth",
                    pd.date_range(
                        "2011-05-20T00:00:25.927", periods=az_len, freq="250ms"
                    ),
                ),
                "longitude": -98.13,
                "latitude": 36.74,
                "altitude": 378,
                "crs_wkt": 0,
                "x": (("azimuth", "range"), x),
                "y": (("azimuth", "range"), y),
                "z": (("azimuth", "range"), z),
            },
            data_vars={
                "DBZH": (
                    (append_dim, "azimuth", "range"),
                    np.random.uniform(-10, 5, size=(1, az_len, rg_len)),
                ),
                "ZDR": (
                    (append_dim, "azimuth", "range"),
                    np.random.uniform(-2, 3, size=(1, az_len, rg_len)),
                ),
                "PHIDP": (
                    (append_dim, "azimuth", "range"),
                    np.random.uniform(0, 360, size=(1, az_len, rg_len)),
                ),
                "RHOHV": (
                    (append_dim, "azimuth", "range"),
                    np.random.uniform(0.2, 1.0, size=(1, az_len, rg_len)),
                ),
                "sweep_mode": (append_dim, ["azimuth_surveillance"]),
                "sweep_number": (append_dim, [0]),
                "prt_mode": (append_dim, ["not_set"]),
                "follow_mode": (append_dim, ["not_set"]),
                "sweep_fixed_angle": (append_dim, [0.4834]),
            },
        )

        return DataTree.from_dict({"/": root_data, "sweep_0": sweep_data})

    return _create_radar_dtree
