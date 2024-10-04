from datetime import datetime

from sigmet2zarr.utils import (
    create_query,
    check_if_exist,
    timer_func,
    batch,
    data_accessor,
)
from sigmet2zarr.task2zarr import prepare2append, dt2zarr2
import fsspec
import xradar as xd
import dask.bag as db
from dask.distributed import Client, LocalCluster


def accessor_wrapper(filename):
    return prepare2append(
        xd.io.open_iris_datatree(data_accessor(filename)),
        append_dim="vcp_time",
        radar_name="GUA",
    )


@timer_func
def radar_convert():
    cluster = LocalCluster(dashboard_address="127.0.0.1:8785")
    client = Client(cluster)
    radar_name = "Guaviare"
    v = 2
    consolidated = False if v == 3 else True
    zarr_store = f"../zarr/{radar_name}_test.zarr"
    year, months, days = 2022, range(6, 7), range(1, 5)  # Guaviare
    # year, months, days = 2022, range(3, 4), range(3, 4)
    for month in months:
        for day in days:
            date_query = datetime(year=year, month=month, day=day)
            query = create_query(date=date_query, radar_site=radar_name)
            str_bucket = "s3://s3-radaresideam/"
            fs = fsspec.filesystem("s3", anon=True)
            radar_files = [
                f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))
            ]
            for files in batch(radar_files, n=12):
                bag = db.from_sequence(files, npartitions=len(files)).map(
                    accessor_wrapper
                )
                ls_dtree = bag.compute()
                for dtree in ls_dtree:
                    dt2zarr2(
                        dtree,
                        zarr_store=zarr_store,
                        zarr_version=v,
                        append_dim="vcp_time",
                        consolidated=consolidated,
                    )


def main():
    radar_convert()


if __name__ == "__main__":
    main()
