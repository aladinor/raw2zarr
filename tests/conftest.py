import gzip
import tempfile

import fsspec
import pytest


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


# --------- Generalized Fixtures for File or File-Like Objects ---------


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    """
    Parameterize whether to use a file path or a file-like object for testing.
    """
    return request.param


@pytest.fixture(scope="session")
def nexradlevel2_files(request):
    return request.getfixturevalue(request.param)
