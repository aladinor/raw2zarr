from datetime import datetime
import time
import zarr
import xarray as xr
from sigmet2zarr.utils import (
    create_query,
    check_if_exist,
    timer_func,
    batch,
    data_accessor,
)
from sigmet2zarr.task2zarr import raw2zarr, prepare2append, dt2zarr2
import fsspec
import xradar as xd
from xarray.backends.api import open_datatree
import s3fs
import dask.bag as db
from dask.distributed import Client, LocalCluster


def consolidated_dt():
    time.time()
    radar_name = "Guaviare"
    v = 2
    path = f"/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/{radar_name}_V{v}.zarr"
    store = zarr.DirectoryStore(path)
    zarr.consolidate_metadata(store)
    # zarr.storage.ConsolidatedMetadataStore(store)


def s3_data():
    URL = "https://js2.jetstream-cloud.org:8001/"
    path = f"pythia/radar/erad2024"
    fs = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=URL))
    file = s3fs.S3Map(f"{path}/zarr_radar/erad_2024.zarr", s3=fs)
    dt = open_datatree(file, engine="zarr", consolidated=True)
    # Open the store using consolidated metadata
    # store = zarr.DirectoryStore(file)
    root = zarr.open(file, mode="r")
    # Access and print the structure of the dataset
    print(root.tree())


@timer_func
def op_dt():
    time.time()
    radar_name = "Guaviare"
    v = 2
    path = f"/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/{radar_name}_V{v}.zarr"
    dt = open_datatree(filename_or_obj=path, engine="zarr", consolidated=True)
    print("done")


@timer_func
def radar_convert():
    radar_name = "Guaviare"
    v = 2
    consolidated = False if v == 3 else True
    zarr_store = f"/media/alfonso/external_ssd/zarr/{radar_name}_V{v}_loop.zarr"
    year, months, days = 2022, range(6, 7), range(1, 2)  # Guaviare
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
            if radar_files:
                for i in radar_files[:10]:
                    exist = check_if_exist(
                        i, path="/media/alfonso/drive/Alfonso/python/raw2zarr/results"
                    )
                    if not exist:
                        raw2zarr(
                            i,
                            zarr_store=zarr_store,
                            mode="a",
                            consolidated=consolidated,
                            append_dim="vcp_time",
                            zarr_version=v,
                            p2c="/media/alfonso/drive/Alfonso/python/raw2zarr/results",
                            # elevation=[0.5]
                        )
            else:
                print(f"mes {month}, dia {day} no tienen datos")
        print(f"mes {month}, dia {day}")
    print("termine")


def accessor_wrapper(filename):
    return prepare2append(
        xd.io.open_iris_datatree(data_accessor(filename)),
        append_dim="vcp_time",
        radar_name="GUA",
    )


@timer_func
def radar_convert2():
    cluster = LocalCluster(dashboard_address="127.0.0.1:8785")
    client = Client(cluster)
    radar_name = "Guaviare"
    v = 2
    consolidated = False if v == 3 else True
    zarr_store = (
        f"/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/{radar_name}_V{v}.zarr"
    )
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
            print(len(radar_files))
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


def zarr_example():
    ds = xr.open_dataset("air.zarr", engine="zarr")
    st = zarr.DirectoryStore("air.zarr")
    root = zarr.open(st, mode="r")
    print(1)
    # ds.to_zarr("air.zarr")


def comp_dict():
    import json
    from deepdiff import DeepDiff

    with open(
        "/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/Guaviare_V2.zarr/.zmetadata"
    ) as f:
        dt1 = json.load(f)

    with open("/zarr/Guaviare_V2.zarr/zmetadata") as f1:
        dt2 = json.load(f1)
    diff = DeepDiff(dt1, dt2)
    print(diff)


def open_erad():
    dtree = xr.backends.api.open_datatree(
        "/media/alfonso/drive/Alfonso/python/zarr_radar/erad_2024.zarr",
        engine="zarr",
        consolidated=True,
        chunks={},
    )

    ds = dtree["cband/sweep_0"].ds
    x = ds["cband/sweep_0"].x
    y = ds["cband/sweep_0"].y
    z = ds["cband/sweep_0"].z

    print(tot)


def main():
    radar_convert2()
    # radar_convert()
    pass


if __name__ == "__main__":
    main()
