import re

import icechunk
import pandas as pd
from xarray import Dataset, DataTree
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

from ..io.preprocess import normalize_input_for_xradar


def get_icechunk_repo(
    zarr_store: str,
    use_manifest_config: bool = True,
) -> icechunk.Repository:
    storage = icechunk.local_filesystem_storage(zarr_store)

    repo_config = None
    if use_manifest_config:
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

    try:
        return icechunk.Repository.create(storage, config=repo_config)
    except icechunk.IcechunkError:
        return icechunk.Repository.open(storage, config=repo_config)


def extract_timestamp(filename: str) -> pd.Timestamp:
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return pd.to_datetime(f"{date_part}{time_part}", format="%Y%m%d%H%M%S")

    match = re.search(r"[A-Z]{3}(\d{6})(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return pd.to_datetime(f"{date_part}{time_part}", format="%y%m%d%H%M%S")

    raise ValueError(f"Could not parse timestamp from filename: {filename}")


def remove_dims(dtree: DataTree, dim: str = "sweep") -> DataTree:
    def remove(ds: Dataset, dim: str = "sweep"):
        try:
            return ds.drop_dims(dim)
        except ValueError:
            return ds

    return dtree.map_over_datasets(remove, dim)


def extract_file_metadata(
    radar_file, engine="nexradlevel2"
) -> tuple[pd.Timestamp, str]:
    """
    Extract both timestamp and VCP number from radar file in single operation.

    More efficient than separate calls since it only reads the file once.

    Note: Error handling is now done at the batch level for distributed processing.

    Parameters:
        radar_file (str): Path to radar file
        engine (str): Radar file engine type

    Returns:
        tuple: (timestamp: pd.Timestamp, vcp_number: int)

    Raises:
        Exception: If file cannot be processed (handled by caller)
    """
    # Extract timestamp from filename (fast regex operation)
    timestamp = extract_timestamp(radar_file)

    # Extract VCP from file header (requires file read)
    if engine == "nexradlevel2":
        vcp_number = NEXRADLevel2File(
            normalize_input_for_xradar(radar_file)
        ).get_msg_5_data()["pattern_number"]
    else:
        raise ValueError(f"Engine not supported: {engine}")

    return timestamp, vcp_number


def _log_problematic_file(filepath: str, error_msg: str, log_file: str = None):
    """
    Log problematic files to output.txt with error details.

    Parameters:
        filepath (str): Path to the problematic file
        error_msg (str): Error message description
        log_file (str): Path to log file. If None, uses "output.txt" in current directory
    """
    import os
    from datetime import datetime

    # Use provided log_file or default to output.txt in current directory
    if log_file is None:
        log_file = "output.txt"

    log_entry = f"{datetime.now().isoformat()}, {filepath}, SKIPPED:, {error_msg}\n"

    # Ensure directory exists
    os.makedirs(
        os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True
    )

    # Write to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Also print to console for immediate feedback
    print(f"Skipping problematic file: {os.path.basename(filepath)} | {error_msg}")
