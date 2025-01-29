import fsspec
from datetime import datetime

from raw2zarr.dtree_builder import append_sequential, append_parallel
from raw2zarr.utils import timer_func, create_query, prepare2read


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
        zs = f"../zarr/{radar_name}.zarr"
        return radar_files, zs, "iris"
    elif engine == "nexradlevel2":
        # NEXRAD
        radar = "KVNX"
        zs = f"../zarr/{radar}_sail.zarr"
        query = f"2012/01/29/{radar}/{radar}"
        str_bucket = "s3://noaa-nexrad-level2/"
        radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
        radar_files = radar_files
        return radar_files, zs, "nexradlevel2"


@timer_func
def main():

    # IRIS Colombia
    radar_files, zs, engine = get_radar_files("iris")

    # NEXRAD
    radar_files, zs, engine = get_radar_files("nexradlevel2")
    # radar_fil es = [
    #     # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_154426_V06',
    #     # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_154815_V06',
    #     # # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155154_V06',
    #     # # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155533_V06',
    #     # # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06',
    #     # 's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06_MDM',
    #     's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_160251_V06',
    #     's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_160643_V06',
    #     's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161058_V06',
    #     's3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06' ### SAIL mode enabled
    # ]
    zarr_format = 2
    append_dim = "vcp_time"
    # append_parallel(
    append_sequential(
        radar_files,  # [1:4],
        # ["s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_124851_V06"],
        zarr_store=zs,
        append_dim=append_dim,
        zarr_format=zarr_format,
        batch_size=10,
        engine=engine,
    )


if __name__ == "__main__":
    main()
