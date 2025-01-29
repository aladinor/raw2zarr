import fsspec
import xradar as xd
import pyart
from xarray import DataTree
import xarray as xr
from time import time
import s3fs


def main():
    print(xr.__version__)
    st = time()
    ## S3 bucket connection
    URL = "https://js2.jetstream-cloud.org:8001/"
    path = f"pythia/radar/erad2024"
    fs = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=URL))
    file = s3fs.S3Map(f"{path}/zarr_radar/Guaviare_test.zarr", s3=fs)

    # opening datatree stored in zarr
    dtree = xr.backends.api.open_datatree(
        file, engine="zarr", consolidated=True, chunks={}
    )
    print(f"total time: {time() -st}")


def open_dtree():
    st = time()
    print(xr.__version__)
    path = "/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/Guaviare_V2.zarr"
    # path = "/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/nexrad2.zarr"
    dt = xr.open_datatree(
        path,
        engine="zarr",
        consolidated=True,
        chunks={},
        # group="sweep_0"
    )
    print(f"total time: {time() -st}")


if __name__ == "__main__":
    # open_dtree()

    file = "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_124851_V06"
    file = "s3://noaa-nexrad-level2/2024/06/29/KILX/KILX20230629_120200_V06"
    local_file = fsspec.open_local(
        f"simplecache::s3://{file}",
        s3={"anon": True},
        filecache={"cache_storage": "."},
    )
    #
    # radar = pyart.io.read_nexrad_archive(local_file)
    dtree = xd.io.open_nexradlevel2_datatree(local_file)
