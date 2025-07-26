from datetime import datetime

import fsspec
import numpy as np
import xarray as xr

from raw2zarr.builder.builder_utils import get_icechunk_repo
from raw2zarr.builder.convert import convert_files
from raw2zarr.utils import create_query, timer_func


def get_radar_files(engine, radar_site="KVNX", start_date="2011/05/20", num_days=2):
    """
    Get radar files for specified number of consecutive days.

    Args:
        engine: "iris" or "nexradlevel2"
        radar_site: Radar site code (e.g., "KVNX", "KILX")
        start_date: Start date in format "YYYY/MM/DD"
        num_days: Number of consecutive days to fetch
    """
    from datetime import datetime, timedelta

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
        radar = radar_site
        zs = f"../zarr/{radar}.zarr"
        str_bucket = "s3://noaa-nexrad-level2/"

        # Generate consecutive dates
        start_dt = datetime.strptime(start_date, "%Y/%m/%d")
        dates = []
        for i in range(num_days):
            date = start_dt + timedelta(days=i)
            dates.append(date.strftime("%Y/%m/%d"))

        radar_files = []
        for date in dates:
            query = f"{date}/{radar}/{radar}"
            day_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
            radar_files.extend(day_files)
            print(f"Found {len(day_files)} files for {date}")

        print(f"Total files for {num_days} days: {len(radar_files)}")
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

    # IRIS Colombia
    # radar_files, zarr_store, engine = get_radar_files("iris")

    # NEXRAD - 2 days of data
    radar_files, zarr_store, engine = get_radar_files(
        engine="nexradlevel2", radar_site="KVNX", start_date="2011/05/20", num_days=2
    )
    # t = load_radar_data(radar_files[0], engine=engine)
    # if dynamic scans
    # radar_files = get_dynamic_scans()

    repo = get_icechunk_repo(zarr_store=zarr_store)

    # radar_files = glob.glob("../data/*")

    zarr_version = 3
    append_dim = "vcp_time"
    convert_files(
        radar_files,
        append_dim=append_dim,
        repo=repo,
        zarr_format=zarr_version,
        engine=engine,
        process_mode="parallel",
        remove_strings=True,
    )


if __name__ == "__main__":
    main()
