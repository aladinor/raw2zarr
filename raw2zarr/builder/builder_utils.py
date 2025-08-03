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


def generate_vcp_samples(
    vcp_time_mapping: dict,
    sample_percentage: float = 15.0,
    output_path: str = None,
    max_samples_per_vcp: int = 300,
) -> dict:
    """
    Generate VCP validation samples from discovered files.

    Args:
        vcp_time_mapping: VCP time mapping dictionary from create_vcp_time_mapping
        sample_percentage: Percentage of files to sample for validation (default 15%)
        output_path: Path to save JSON file. If None, doesn't save to file
        max_samples_per_vcp: Maximum number of samples per VCP pattern

    Returns:
        Dictionary of VCP samples
    """
    import random
    import json
    import os

    print("🔍 Generating VCP validation samples:")

    # Build VCP samples dictionary
    vcp_samples = {}
    for vcp_name, vcp_info in vcp_time_mapping.items():
        time_span = vcp_info["timestamps"][-1] - vcp_info["timestamps"][0]
        all_files = vcp_info["files"]

        # Calculate number of samples based on percentage
        percentage_samples = int(len(all_files) * (sample_percentage / 100))
        num_samples = min(max_samples_per_vcp, percentage_samples)
        num_samples = max(1, num_samples)  # At least 1 sample

        sample_files = random.sample(all_files, num_samples)

        # Extract just the file paths for JSON
        vcp_samples[vcp_name] = [file_info["filepath"] for file_info in sample_files]

        print(f"  🔹 {vcp_name}: {vcp_info['file_count']} files ({time_span})")
        print(f"     📄 Sampled {len(sample_files)} files ({sample_percentage:.1f}%)")

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(vcp_samples, f, indent=2)

        print(f"✅ Written VCP samples to {output_path}")

    print(f"📊 Total VCP patterns: {len(vcp_samples)}")
    total_samples = sum(len(samples) for samples in vcp_samples.values())
    print(f"📁 Total validation samples: {total_samples}")

    return vcp_samples


def extract_single_metadata(file_info):
    """Extract metadata from a single file - optimized for Client.map()"""
    original_index, file = file_info
    try:
        from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

        from raw2zarr.io.preprocess import normalize_input_for_xradar

        timestamp = extract_timestamp(file)

        # Extract VCP from file header (requires file read)
        vcp_number = NEXRADLevel2File(
            normalize_input_for_xradar(file)
        ).get_msg_5_data()["pattern_number"]

        return original_index, file, (timestamp, vcp_number)

    except Exception as e:
        return original_index, file, ("ERROR", f"Metadata extraction failed: {str(e)}")


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
