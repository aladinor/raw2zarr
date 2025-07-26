"""
Python script to ingest ERA5 data from a specified URL and save it to a local directory.
"""

import os
from rich.console import Console
import icechunk
import asyncio
from datetime import datetime, timedelta
import fsspec
import re
from pathlib import Path
from raw2zarr.builder.convert import convert_files
from raw2zarr.builder.builder_utils import get_icechunk_repo

console = Console()

# Conditional imports for cloud mode
try:
    import coiled
    import arraylake as al
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

str_bucket = "s3://noaa-nexrad-level2/"

def create_cluster(worker_count: int, region: str):
    workspace = "earthmover-devs"
    token_em = "ema_40f07a24d1f94d369542add79a4d372a_1140265c99a694346ee3b8e669adcaa278c764c60f94b56dfa792a85bd17806b"
    cluster_name = f"nexrad-{os.environ['USER']}-{region}"
    
    cluster = coiled.Cluster(
        name=cluster_name,
        asynchronous=False,
        region=region,
        workspace=workspace,
        software="nexrad-env",
        environ={"ARRAYLAKE_TOKEN": token_em, "ICECHUNK_LOG": "icechunk=info"},
        worker_cpu=2,
        worker_memory="8GiB",
        credentials=None,
        n_workers=(50, worker_count),
    )
    return cluster


async def get_arraylake_repo(repo_name: str):
    """Open an Arraylake Icechunk repo with custom manifest split config."""
    if not CLOUD_AVAILABLE:
        raise ImportError("Cloud dependencies (coiled, arraylake) not available")
    
    token_em = "ema_40f07a24d1f94d369542add79a4d372a_1140265c99a694346ee3b8e669adcaa278c764c60f94b56dfa792a85bd17806b"
    client = al.Client(token=token_em)
    aclient = client.aclient

    split_config = icechunk.ManifestSplittingConfig.from_dict(
        {
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName(
                    "vcp_time"
                ): 65000  # ~1 year at mixed intervals
            }
        }
    )

    var_condition = icechunk.ManifestPreloadCondition.name_matches(
        r"^(vcp_time|azimuth|range|x|y|z)$"
    )
    size_condition = icechunk.ManifestPreloadCondition.num_refs(
        0, 100
    )  # Small arrays

    preload_if = icechunk.ManifestPreloadCondition.and_conditions(
        [var_condition, size_condition]
    )

    preload_config = icechunk.ManifestPreloadConfig(
        max_total_refs=1000,
        preload_if=preload_if,
    )

    repo_config = icechunk.RepositoryConfig(
        manifest=icechunk.ManifestConfig(
            splitting=split_config, preload=preload_config
        ),
    )
    repo = await aclient.get_repo(repo_name, config=repo_config)
    return repo


def parse_nexrad_filename(fname):
    """Extract datetime from filename: e.g., KVNX20110520_000238_V03.gz"""
    match = re.search(r"(\d{8})_(\d{6})", fname)
    if match:
        date_str = match.group(1) + match.group(2)
        return datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return None


async def list_day_files(fs, date, radar):
    prefix = f"{str_bucket}{date.strftime('%Y/%m/%d')}/{radar}/{radar}"
    files = await asyncio.to_thread(fs.glob, f"{prefix}*")
    return [f"s3://{f}" for f in files]


async def get_radar_files_async(radar_site="KVNX", start_date=None, num_days=None,
                                start_time=None, end_time=None):
    """
    List radar files from NEXRAD S3 bucket within a time range.

    Either use:
        - start_date (YYYY/MM/DD) + num_days
        - start_time + end_time (datetime objects)
    """
    fs = fsspec.filesystem("s3", anon=True)
    zs = f"earthmover/{radar_site}"

    # Determine time range
    if start_time and end_time:
        start_dt, end_dt = start_time, end_time
    elif start_date and num_days:
        start_dt = datetime.strptime(start_date, "%Y/%m/%d")
        end_dt = start_dt + timedelta(days=num_days)
    else:
        raise ValueError("Provide either (start_date + num_days) or (start_time + end_time)")

    # Generate list of days to query
    days = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]

    # Parallel file listing
    tasks = [list_day_files(fs, d, radar_site) for d in days]
    results = await asyncio.gather(*tasks)
    all_files = [f for sublist in results for f in sublist]

    # Filter and sort files within exact time range
    filtered = [
        (f, dt) for f in all_files
        if (dt := parse_nexrad_filename(f)) and start_dt <= dt <= end_dt
    ]

    filtered_sorted = [f for f, _ in sorted(filtered, key=lambda x: x[1])]

    print(f"Found {len(filtered_sorted)} radar files between {start_dt} and {end_dt}")
    return filtered_sorted, zs, "nexradlevel2"


def main(local_mode: bool = True):
    """
    Run radar data ingestion pipeline.
    
    Args:
        local_mode (bool): If True, run locally with LocalCluster and local zarr store.
                          If False, use Coiled cluster and Arraylake repository.
    """
    # Configuration
    zarr_version = 3
    append_dim = "vcp_time"
    
    # Mode-specific settings
    if local_mode:
        print("🏠 Running in LOCAL MODE")
        target_repo = "zarr/kvnxtest"
        process_mode = "parallel"  # Use local parallel processing
        worker_count = None  # Let LocalCluster decide
        region = None
    else:
        print("☁️  Running in CLOUD MODE")
        if not CLOUD_AVAILABLE:
            raise ImportError("Cloud mode requires 'coiled' and 'arraylake' packages")
        target_repo = "earthmover/KVNX"
        process_mode = "parallel"
        worker_count = 200
        region = "us-east-1"

    # Get radar files
    with console.status(f"Getting Radar Files"):
        files, zs, engine = asyncio.run(get_radar_files_async(
            radar_site="KVNX",
            start_time=datetime(2011, 5, 1, 0, 00, 0),  # 30 min before corrupted file
            end_time=datetime(2011, 5, 31, 23, 00, 0)    # 30 min after corrupted file
            )
        )

    # Get repository
    if local_mode:
        with console.status(f"Setting up local repository {target_repo}"):
            repo = get_icechunk_repo(target_repo)
    else:
        with console.status(f"Getting Arraylake repository {target_repo}"):
            repo = asyncio.run(get_arraylake_repo(target_repo))

    # Set up cluster
    if local_mode:
        cluster = None  # Let convert_files use LocalCluster
        print("🖥️  Using local Dask cluster")
    else:
        with console.status(f"Creating Coiled cluster"):
            cluster = create_cluster(worker_count, region)
        print(f"☁️  Using Coiled cluster with {worker_count} workers")

    # Run ingestion
    with console.status(f"Ingesting {len(files)} files"):
        convert_files(
            files,
            append_dim=append_dim,
            repo=repo,
            zarr_format=zarr_version,
            engine=engine,
            process_mode=process_mode,
            remove_strings=True,
            cluster=cluster
        )
    
    if local_mode:
        print(f"✅ Local data saved to: {Path(target_repo).resolve()}")
    else:
        print(f"✅ Cloud data saved to: {target_repo}")

if __name__ == '__main__':
    main(local_mode=False)
    # main(local_mode=True)

