from datetime import datetime

from raw2zarr.utils import (
    create_query,
    timer_func,
    data_accessor,
)
from raw2zarr.task2zarr import prepare2append
from raw2zarr.dtree_builder import datatree_builder
import fsspec
import xradar as xd



def accessor_wrapper(filename):
    return prepare2append(
        xd.io.open_iris_datatree(data_accessor(filename)),
        append_dim="vcp_time",
        radar_name="GUA",
    )


@timer_func
def radar_convert():
    radar_name = "Guaviare"
    year, month, day = 2022, 6, 1  # Guaviare

    date_query = datetime(year=year, month=month, day=day)
    query = create_query(date=date_query, radar_site=radar_name)
    str_bucket = "s3://s3-radaresideam/"
    fs = fsspec.filesystem("s3", anon=True)
    radar_files = [
        f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))
    ][:100]

    builder = datatree_builder(radar_files, batch_size=4)
    print(builder)

def main():
    radar_convert()


if __name__ == "__main__":
    main()
