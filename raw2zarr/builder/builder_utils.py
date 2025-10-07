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

    match = re.search(r"(\d{10})_(\d{2})", filename)
    if match:
        date_part, hour_part = match.groups()
        return pd.to_datetime(f"{date_part}{hour_part}", format="%Y%m%d%H%M")

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
    import json
    import os
    import random

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


def extract_sweep_time(msg_31_header_entry):
    """
    Extract timestamp from msg_31_header entry.

    Uses the formula from metadata_extractor.py:
        date = (collect_date - 1) * 86400e3
        milli = collect_ms
        time_ms = date + milli
        timestamp = to_datetime(time_ms, unit='ms', utc=True)

    Parameters
    ----------
    msg_31_header_entry : dict
        Single entry from msg_31_header array (first or last azimuth)

    Returns
    -------
    pd.Timestamp
        Timestamp for this sweep
    """
    date = (msg_31_header_entry["collect_date"] - 1) * 86400e3
    milli = msg_31_header_entry["collect_ms"]
    time_ms = date + milli
    return pd.to_datetime(time_ms, unit="ms", utc=True)


def detect_temporal_slices(msg_31_headers, elevation_angles, scan_type):
    """
    Split sweeps into temporal slices based on SAILS/MRLE restart patterns.

    For STANDARD and AVSET scans, returns a single slice with all sweeps.
    For SAILS/MESO-SAILS/MRLE scans, detects restart points where 0.5° appears
    after higher elevations, indicating a new temporal slice.

    Parameters
    ----------
    msg_31_headers : list
        List of msg_31_header arrays (one per sweep)
    elevation_angles : list of float
        List of elevation angles (may include None values)
    scan_type : str
        Scan type classification: "STANDARD", "SAILS", "MESO-SAILS×N", "MRLE×N", "AVSET"

    Returns
    -------
    list of dict
        List of temporal slice dictionaries, each containing:
        - slice_id: int (0, 1, 2, ...)
        - sweep_indices: list of int (indices into msg_31_headers)
        - start_time: pd.Timestamp
        - end_time: pd.Timestamp
        - scan_type: str (same as input scan_type)
    """
    if scan_type == "STANDARD" or scan_type == "AVSET":
        # Single slice with all sweeps
        return [
            {
                "slice_id": 0,
                "sweep_indices": list(range(len(msg_31_headers))),
                "start_time": extract_sweep_time(msg_31_headers[0][0]),
                "end_time": extract_sweep_time(msg_31_headers[-1][-1]),
                "scan_type": scan_type,
            }
        ]

    # For SAILS/MRLE, detect restart points
    slices = []
    start_idx = 0
    slice_id = 0

    # Round elevations for comparison (0.48 → 0.5)
    # Filter out None values
    rounded_elevs = [round(e, 1) if e is not None else None for e in elevation_angles]

    for i in range(1, len(rounded_elevs)):
        if rounded_elevs[i] is None or rounded_elevs[i - 1] is None:
            continue

        # Restart point: 0.5° appears after we've moved past it
        # Example: [..., 0.9, 0.9, 0.5, 0.5, ...] or [..., 1.3, 1.8, 2.4, 0.5, 0.5, ...]
        #                          ↑ Restart here           ↑ Restart here
        if rounded_elevs[i] <= 0.6 and rounded_elevs[i - 1] > 0.6:
            # Close current slice
            slices.append(
                {
                    "slice_id": slice_id,
                    "sweep_indices": list(range(start_idx, i)),
                    "start_time": extract_sweep_time(msg_31_headers[start_idx][0]),
                    "end_time": extract_sweep_time(msg_31_headers[i - 1][-1]),
                    "scan_type": scan_type,
                }
            )
            start_idx = i
            slice_id += 1

    # Add final slice
    slices.append(
        {
            "slice_id": slice_id,
            "sweep_indices": list(range(start_idx, len(rounded_elevs))),
            "start_time": extract_sweep_time(msg_31_headers[start_idx][0]),
            "end_time": extract_sweep_time(msg_31_headers[-1][-1]),
            "scan_type": scan_type,
        }
    )

    return slices


def extract_single_metadata(file_info, engine="iris"):
    """
    Extract metadata from a single file - optimized for Client.map().

    For dynamic NEXRAD scans (SAILS, MRLE), returns multiple entries (one per temporal slice).
    For standard scans, returns single entry.

    Parameters
    ----------
    file_info : tuple
        (original_index, file_path) tuple
    engine : str
        Engine type: "nexradlevel2", "iris", or "odim"

    Returns
    -------
    list of tuple
        List of tuples, each containing:
        (original_index, file, timestamp, vcp, slice_id, sweep_indices, scan_type, elevation_angles)

        For STANDARD scans: returns single-entry list
        For MESO-SAILS×3: returns 4-entry list (one per temporal slice)

        elevation_angles: List of elevations for sweeps in this slice (for NEXRAD) or None (for IRIS/ODIM)
    """
    original_index, file = file_info

    try:
        if engine == "nexradlevel2":
            # Local import to avoid circular dependency
            from ..utils.metadata_extractor import classify_dynamic_type

            nex = NEXRADLevel2File(normalize_input_for_xradar(file))
            msg_5 = nex.get_msg_5_data()
            vcp_number = msg_5["pattern_number"]
            vcp = f"VCP-{vcp_number}"

            # Extract elevation angles and sweep counts
            n_groups = len(nex.msg_31_header)
            elevation_angles = []
            for group in range(n_groups):
                try:
                    elev = round(msg_5["elevation_data"][group]["elevation_angle"], 2)
                    elevation_angles.append(elev)
                except Exception:
                    elevation_angles.append(None)

            # Get expected sweeps
            exp_sweeps = msg_5.get("number_elevation_cuts")
            act_sweeps = len(nex.msg_31_header)  # Actual sweeps from header

            # Get data sweeps - actual sweeps present in file data
            data_sweeps = list(nex.data.keys())

            # Check for corrupted files (data sweeps don't match header)
            # Sweeps should be consecutive integers from 0 to act_sweeps-1
            expected_keys = set(range(act_sweeps))
            actual_keys = set(data_sweeps)

            if actual_keys != expected_keys:
                # File is corrupted - missing or extra sweep indices
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                error_parts = []
                if missing:
                    error_parts.append(f"missing sweeps {missing}")
                if extra:
                    error_parts.append(f"extra sweeps {extra}")
                error_msg = f"Corrupted: {', '.join(error_parts)}"
                return [
                    (original_index, file, "ERROR", error_msg, 0, [], "CORRUPTED", [])
                ]

            # Classify dynamic scan type
            classification = classify_dynamic_type(
                elevation_angles=elevation_angles,
                act_sweeps=act_sweeps,
                exp_sweeps=exp_sweeps,
            )
            scan_type = classification["dynamic_type"]

            # Detect temporal slices
            slices = detect_temporal_slices(
                nex.msg_31_header, elevation_angles, scan_type
            )

            # Return list of entries (one per temporal slice)
            results = []
            for slice_info in slices:
                # Extract elevation angles for this slice's sweep indices
                slice_elevations = [
                    elevation_angles[i] for i in slice_info["sweep_indices"]
                ]
                results.append(
                    (
                        original_index,
                        file,
                        slice_info["start_time"],  # Timestamp
                        vcp,  # VCP name
                        slice_info["slice_id"],  # 0, 1, 2, 3 for MESO-SAILS×3
                        slice_info[
                            "sweep_indices"
                        ],  # [0,1,2,3,4,5,6,7] for first slice
                        slice_info["scan_type"],  # "MESO-SAILS×3" | "STANDARD"
                        slice_elevations,  # [0.5, 0.5, 0.9, 0.9, ...] for this slice
                    )
                )

            return results

        elif engine == "iris":
            from xradar.io.backends.iris import _check_iris_file

            _file = normalize_input_for_xradar(file)
            sid, opener = _check_iris_file(_file)
            with opener(_file, loaddata=False) as ds:
                vcp_number = ds.product_hdr["product_configuration"]["task_name"]
                timestamp = extract_timestamp(file)
                # Return single entry wrapped in list for consistency
                return [
                    (
                        original_index,
                        file,
                        timestamp,
                        vcp_number.strip(),
                        0,
                        None,
                        "STANDARD",
                        None,  # elevation_angles - not needed for IRIS
                    )
                ]

        elif engine == "odim":
            import h5netcdf

            timestamp = extract_timestamp(file)
            with h5netcdf.File(file, "r", decode_vlen_strings=True) as fh:
                if "scan_name" in fh.attrs:
                    vcp_number = fh.attrs["scan_name"]
                    if isinstance(vcp_number, bytes):
                        vcp_number = vcp_number.decode("utf-8")
                    return [
                        (
                            original_index,
                            file,
                            timestamp,
                            vcp_number,
                            0,
                            None,
                            "STANDARD",
                            None,
                        )
                    ]
                else:
                    possible_attrs = ["what/object", "what/source", "how/task"]
                    for attr_path in possible_attrs:
                        if attr_path in fh.attrs:
                            vcp_number = fh.attrs[attr_path]
                            if isinstance(vcp_number, bytes):
                                vcp_number = vcp_number.decode("utf-8")
                            return [
                                (
                                    original_index,
                                    file,
                                    timestamp,
                                    vcp_number,
                                    0,
                                    None,
                                    "STANDARD",
                                    None,  # elevation_angles - not needed for ODIM
                                )
                            ]
                    return [
                        (
                            original_index,
                            file,
                            timestamp,
                            "DEFAULT",
                            0,
                            None,
                            "STANDARD",
                            None,
                        )
                    ]

    except Exception as e:
        return [
            (
                original_index,
                file,
                "ERROR",
                f"Metadata extraction failed: {str(e)}",
                0,
                None,
                None,
                None,
            )
        ]


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
