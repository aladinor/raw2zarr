import asyncio
import glob
import os
import shutil
import time
from datetime import datetime

import boto3
import fsspec
import icechunk
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster

from raw2zarr.builder.builder_utils import get_icechunk_repo
from raw2zarr.builder.convert import convert_files
from raw2zarr.utils import create_query, load_vcp_samples
from raw2zarr.utils.core import get_radar_files_async


def remove_folder_if_exists(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Folder '{path}' has been removed.")
    else:
        print(f"Folder '{path}' does not exist.")


# Example usage


def get_radar_files(engine):
    fs = fsspec.filesystem("s3", anon=True)

    if engine == "iris":
        radar_name = "Guaviare"
        year, month, day = 2025, 9, 5
        date_query = datetime(year=year, month=month, day=day)
        str_bucket = "s3://s3-radaresideam/"
        query = create_query(date=date_query, site=radar_name)
        radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
        zs = f"../zarr/{radar_name}"
        vcp_config_file = "ideam.json"
        return radar_files, zs, "iris", vcp_config_file
    elif engine == "nexradlevel2":
        # NEXRAD
        radar = "KVNX"
        zs = f"../zarr/{radar}"
        query = f"2011/05/20/{radar}/{radar}"
        str_bucket = "s3://unidata-nexrad-level2/"
        radar_files = [f"s3://{i}" for i in sorted(fs.glob(f"{str_bucket}{query}*"))]
        radar_files = radar_files
        vcp_config_file = "vcp_nexrad.json"
        return radar_files, zs, "nexradlevel2", vcp_config_file
    elif engine == "odim":
        radar = "CASET"
        zs = f"../zarr/{radar}"
        radar_files = sorted(
            glob.glob(
                f"//media/alfonso/drive/Alfonso/python/raw2zarr/data/ECCC/*_{radar}.h5"
            )
        )
        corrupted_files = [
            "2024080602_42_ODIMH5_PVOL6S_VOL_CASSM.h5",
            "2024080602_54_ODIMH5_PVOL6S_VOL_CASSM.h5",
            "2024080603_18_ODIMH5_PVOL6S_VOL_CASSM.h5",
            "2024080603_24_ODIMH5_PVOL6S_VOL_CASSM.h5",
        ]
        radar_files = [
            file for file in radar_files if file.split("/")[-1] not in corrupted_files
        ]
        vcp_config_file = "eccc.json"
        return radar_files, zs, "odim", vcp_config_file
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
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_154426_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_154815_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_155154_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_155533_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_155912_V06_MDM",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_160251_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_160643_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_161058_V06",
        "s3://unidata-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06",  ### SAIL mode enabled
        "s3://unidata-nexrad-level2/2025/02/11/KFCX/KFCX20250211_164314_V06",  ## AVSET + sailsx1 VCP215
        "s3://unidata-nexrad-level2/2025/02/11/KFSX/KFSX20250211_164159_V06",  # AVSET + base tilt (-0.2)
    ]
    radar_files = asyncio.run(
        get_radar_files_async(
            radar_site="KLOT",
            start_time=datetime(2025, 3, 15, 1, 10),
            end_time=datetime(2025, 3, 15, 12, 20),
        )
    )
    return radar_files


def files_with_shape_mismatch(vcp: str = "VCP-21"):
    files = load_vcp_samples(
        "/media/alfonso/drive/Alfonso/python/raw2zarr/data/vcp_samples.json"
    )[vcp]
    return sorted(files), f"../zarr/{vcp}test", "nexradlevel2"


def get_repo_config():
    split_config = icechunk.ManifestSplittingConfig.from_dict(
        {
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName("vcp_time"): 12
                * 24
                * 365  # roughly one year of radar data
            }
        }
    )

    var_condition = icechunk.ManifestPreloadCondition.name_matches(
        r"^(vcp_time|azimuth|range|x|y|z)$"
    )
    size_condition = icechunk.ManifestPreloadCondition.num_refs(0, 100)  # Small arrays

    preload_if = icechunk.ManifestPreloadCondition.and_conditions(
        [var_condition, size_condition]
    )

    preload_config = icechunk.ManifestPreloadConfig(
        max_total_refs=1000,
        preload_if=preload_if,
    )

    return icechunk.RepositoryConfig(
        manifest=icechunk.ManifestConfig(
            splitting=split_config, preload=preload_config
        ),
    )


def get_cluster():
    dashboard_address = "127.0.0.1:8785"
    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit="10GB")
    return cluster


def main():
    # IRIS Colombia
    # radar_files, zarr_store, engine, vcp_config_file = get_radar_files("iris")
    # NEXRAD
    _, zarr_store, engine, vcp_config_file = get_radar_files("nexradlevel2")
    radar_files = get_dynamic_scans()

    # ECCC
    # radar_files, zarr_store, engine, vcp_config_file = get_radar_files("odim")
    # t = load_radar_data(radar_files[0], engine=engine)
    # if dynamic scans
    # radar_files = get_dynamic_scans()
    # radar_files, zarr_store, engine = files_with_shape_mismatch("VCP-32")
    # Choose local or remote Icechunk repository
    use_remote_repo = True

    if use_remote_repo:
        print("Using Remote repo")
        session = boto3.Session(profile_name="osn-nexrad")
        credentials = session.get_credentials()
        access_key = credentials.access_key
        secret_key = credentials.secret_key
        region = "us-east-1"
        endpoint_url = "https://umn1.osn.mghpcc.org"

        remote_prefix = "KLOT"
        repo = get_icechunk_repo(
            zarr_store=remote_prefix,
            local_repo=False,
            bucket="nexrad-arco",
            endpoint_url=endpoint_url,
            region=region,
            access_key=access_key,
            secret_access=secret_key,
            force_path_style=True,
        )
    else:
        print("Using local repo")
        remove_folder_if_exists(zarr_store)
        repo = get_icechunk_repo(zarr_store=zarr_store)

    process_mode = "parallel"
    if process_mode == "parallel":
        cluster = get_cluster()
    else:
        cluster = None

    zarr_version = 3
    append_dim = "vcp_time"

    start = time.time()
    convert_files(
        radar_files[:2],
        append_dim=append_dim,
        repo=repo,
        zarr_format=zarr_version,
        engine=engine,
        process_mode=process_mode,
        remove_strings=True,
        cluster=cluster,
        log_file=f"/media/alfonso/drive/Alfonso/python/raw2zarr/{zarr_store.split('/')[-1]}.txt",
        vcp_config_file=vcp_config_file,
    )
    elapsed = time.time() - start
    print(f"convert_files executed in {elapsed:.4f}s")


if __name__ == "__main__":
    main()
