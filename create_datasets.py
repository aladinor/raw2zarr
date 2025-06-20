import fsspec
import xarray as xr

from raw2zarr.builder.builder_utils import get_icechunk_repo
from raw2zarr.builder.convert import convert_files
from raw2zarr.utils import timer_func


@timer_func
def create_ideam_dt():
    radar = "Guaviare"
    append_dim = "vcp_time"
    engine = "iris"
    zarr_format = 3
    consolidated = True if zarr_format == 2 else False
    zarr_store_ic = f"zarr/{radar}_IC.zarr"
    zarr_store = f"zarr/{radar}.zarr"
    query = f"2025/06/19/{radar}/{radar[:3].upper()}"
    str_bucket = "s3://s3-radaresideam/l2_data"
    fs = fsspec.filesystem("s3", anon=True)
    radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}/{query}*"))]
    repo = get_icechunk_repo(zarr_store=zarr_store_ic)
    convert_files(
        radar_files[-300:],
        append_dim=append_dim,
        repo=repo,
        zarr_store=zarr_store,
        zarr_format=zarr_format,
        engine=engine,
        process_mode="parallel",
        remove_strings=True,
        consolidated=consolidated,
    )
    session = repo.readonly_session("main")
    dtree = xr.open_datatree(
        session.store,
        engine="zarr",
        zarr_format=3,
        consolidated=False,
        chunks={},
        mode="r",
    )
    dtree.to_zarr(
        zarr_store,
        zarr_format=3,
        consolidated=False,
    )
    import shutil

    shutil.rmtree(zarr_store_ic)


@timer_func
def create_nexrad_dt():
    radar = "KVNX"
    append_dim = "vcp_time"
    engine = "nexradlevel2"
    zarr_format = 3
    consolidated = True if zarr_format == 2 else False
    zarr_store_ic = f"zarr/{radar}_IC.zarr"
    zarr_store = f"zarr/{radar}.zarr"
    query = f"2011/05/20/{radar}/{radar}"
    str_bucket = "s3://noaa-nexrad-level2/"
    fs = fsspec.filesystem("s3", anon=True)
    radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
    repo = get_icechunk_repo(zarr_store=zarr_store_ic)
    convert_files(
        radar_files[130:170],
        append_dim=append_dim,
        repo=repo,
        zarr_store=zarr_store,
        zarr_format=zarr_format,
        engine=engine,
        process_mode="parallel-region",
        remove_strings=True,
        consolidated=consolidated,
    )
    session = repo.readonly_session("main")
    dtree = xr.open_datatree(
        session.store,
        engine="zarr",
        zarr_format=3,
        consolidated=False,
        chunks={},
        mode="r",
    )
    dtree.to_zarr(
        zarr_store,
        zarr_format=3,
        consolidated=False,
    )
    import shutil

    shutil.rmtree(zarr_store_ic)


def main():
    create_nexrad_dt()
    create_ideam_dt()
    print(1)


if __name__ == "__main__":
    main()
