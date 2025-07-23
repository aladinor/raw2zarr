import glob
from datetime import datetime

import fsspec
import numpy as np
import xarray as xr

from raw2zarr.builder.convert import convert_files
from raw2zarr.utils import create_query, timer_func
from raw2zarr.builder.builder_utils import get_icechunk_repo


def get_radar_files(engine):

    fs = fsspec.filesystem("s3", anon=True)

    if engine == "iris":
        radar_name = "Guaviare"
        year, month, day = 2022, 6, 1
        date_query = datetime(year=year, month=month, day=day)
        str_bucket = "s3://s3-radaresideam/"
        query = create_query(date=date_query, radar_site=radar_name)
        radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))][
            550:570
        ]
        zs = f"../zarr/{radar_name}2.zarr"
        return radar_files, zs, "iris"
    elif engine == "nexradlevel2":
        # NEXRAD
        radar = "KVNX"
        zs = f"../zarr/{radar}.zarr"
        query = f"2011/05/20/{radar}/{radar}"
        str_bucket = "s3://noaa-nexrad-level2/"
        radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
        radar_files = radar_files
        return radar_files, zs, "nexradlevel2"
    return None


def create_dtree():
    ds_a = xr.Dataset(
        {
            "A": (("x", "y"), np.ones((128, 256))),
        }
    )
    ds_b = xr.Dataset({"B": (("y", "x"), np.ones((256, 128)) * 2)})
    ds_d = xr.Dataset({"G": (("x", "y"), np.zeros((128, 256)))})
    ds_rt = xr.Dataset(
        {"z": (("x", "y"), np.zeros((128, 256))), "w": (("x"), np.ones(128))}
    )

    dt = xr.DataTree.from_dict(
        {"/": ds_rt, "/a": ds_a, "/b": ds_b, "/c": ds_rt, "/c/d": ds_d}
    )
    return dt


def get_dynamic_scans() -> list[str]:
    radar_files = [
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_154426_V06",
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_154815_V06",
        # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155154_V06',
        # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155533_V06',
        # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06',
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06_MDM",
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_160251_V06",
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_160643_V06",
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161058_V06",
        "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06",  ### SAIL mode enabled
        "s3://noaa-nexrad-level2/2025/02/11/KFCX/KFCX20250211_164314_V06",  ## AVSET + sailsx1 VCP215
        "s3://noaa-nexrad-level2/2025/02/11/KFSX/KFSX20250211_164159_V06",  # AVSET + base tilt (-0.2)
    ]
    # radar_files = ["s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_124851_V06"],

    return radar_files


@timer_func
def main():
    import shutil

    # IRIS Colombia
    # radar_files, zarr_store, engine = get_radar_files("iris")

    # NEXRAD
    radar_files, zarr_store, engine = get_radar_files("nexradlevel2")
    # t = load_radar_data(radar_files[0], engine=engine)
    # if dynamic scans
    # radar_files = get_dynamic_scans()

    repo = get_icechunk_repo(zarr_store=zarr_store)

    # radar_files = glob.glob("../data/*")

    zarr_version = 3
    append_dim = "vcp_time"
    convert_files(
        radar_files[215:225],  # Just 2 files for quick test
        append_dim=append_dim,
        repo=repo,
        zarr_format=zarr_version,
        engine=engine,
        process_mode="parallel-region",
        remove_strings=True,
    )


if __name__ == "__main__":
    main()
