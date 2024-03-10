from datetime import datetime
import fsspec
import time
from sigmet2zarr.utils import create_query, check_if_exist
from sigmet2zarr.task2zarr import raw2zarr


def main():
    radar_name = "Carimagua"
    v = 2
    con = False if v == 3 else True
    zarr_store = f"/media/alfonso/drive/Alfonso/python/zarr_radar/{radar_name}_{v}.zarr"
    year, months, days = 2022, range(8, 9), range(9, 13)
    # year, months, days = 2022, range(3, 4), range(3, 4)
    for month in months:
        for day in days:
            date_query = datetime(year=year, month=month, day=day)
            query = create_query(date=date_query, radar_site=radar_name)
            str_bucket = "s3://s3-radaresideam/"
            fs = fsspec.filesystem("s3", anon=True)
            radar_files = sorted(fs.glob(f"{str_bucket}{query}*"))
            if radar_files:
                start_time = time.monotonic()
                for i in radar_files:
                    exist = check_if_exist(i)
                    if not exist:
                        raw2zarr(
                            i,
                            store=zarr_store,
                            mode="a",
                            consolidated=con,
                            append_dim="vcp_time",
                            zarr_version=v,
                            # elevation=[0.5]
                        )
                print(f"Run time for single{time.monotonic() - start_time} seconds")
                print("Done!!!")
            else:
                print(f"mes {month}, dia {day} no tienen datos")
        print(f"mes {month}, dia {day}")
    print("termine")
    pass


if __name__ == "__main__":
    main()
