#!/usr/bin/env python3
"""
NEXRAD Metadata Extractor

Extract comprehensive metadata from NEXRAD Level 2 radar files including:
- VCP information and sweep structure
- Dynamic scan mode detection (AVSET, SAILS, MRLE)
- Per-sweep timing information
- File status classification (valid, dynamic, corrupted)

Usage:
    # Run as script with default settings
    python -m raw2zarr.utils.metadata_extractor

    # Or import and use functions
    from raw2zarr.utils.metadata_extractor import extract_vcp_and_shapes, classify_dynamic_type

    # Extract metadata from single file
    metadata = extract_vcp_and_shapes("s3://path/to/file")

    # Batch process files
    results = process(file_list, output_csv="metadata.csv")
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
from pandas import to_datetime
from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

from ..builder.builder_utils import extract_timestamp
from ..config.utils import load_json_config
from ..io.preprocess import normalize_input_for_xradar

# Dask (optional)
try:
    from dask.distributed import Client, LocalCluster, as_completed

    DASK_AVAILABLE = True
except Exception:
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    as_completed = None  # type: ignore
    DASK_AVAILABLE = False


# =====================
# CONFIG (edit here)
# =====================
USE_DASK = True
DASHBOARD_PORT = 8785  # dashboard at http://127.0.0.1:8785/status
OUTPUT_CSV: str | None = "nexrad_metadata.csv"  # CSV output file
WRITE_PARQUET: bool = True
PARQUET_ROOT: str = "shape_registry"
INCREMENTAL_CSV: bool = True  # Write CSV incrementally for large datasets (>10K files)

# Optional fields (can be disabled to reduce output file size)
INCLUDE_SHAPE_SIGNATURE: bool = (
    False  # SHA-1 hash of sweep structure (for deduplication)
)
INCLUDE_SWEEP_SHAPES_JSON: bool = (
    False  # Detailed per-sweep moment dimensions (large field)
)


def _parse_site_from_path(path: str) -> str | None:
    """Extract site name from file path."""
    try:
        return path.split("/")[-2] or None
    except Exception:
        return None


def _compute_signature(shape: dict[str, Any]) -> str:
    """
    Compute SHA-1 hash signature of sweep structure for deduplication.

    Parameters
    ----------
    shape : Dict[str, Any]
        Sweep structure dictionary

    Returns
    -------
    str
        SHA-1 hexdigest of normalized sweep structure
    """

    def sweep_idx(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    norm: list[tuple[int, int, list[tuple[str, int]]]] = []
    for sk in sorted(shape.keys(), key=sweep_idx):
        s = shape[sk]
        az = int(s.get("az", 0))
        pairs: list[tuple[str, int]] = []
        for kk, vv in s.items():
            if kk == "az" or not isinstance(vv, (int, float)):
                continue
            if kk.endswith("/ngates"):
                moment = kk.split("/")[0]
                try:
                    pairs.append((moment, int(vv)))
                except Exception:
                    pass
        pairs.sort()
        norm.append((sweep_idx(sk), az, pairs))

    payload = json.dumps(norm, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def compare_with_vcp_config(
    azimuth_actual: list[int | None],
    range_actual: list[int | None],
    vcp_number: int,
) -> dict[str, Any]:
    """
    Compare actual sweep dimensions against VCP configuration.

    Parameters
    ----------
    azimuth_actual : List[Optional[int]]
        Actual azimuth values per sweep
    range_actual : List[Optional[int]]
        Actual range gate values per sweep
    vcp_number : int
        VCP number (e.g., 12 for VCP-12)

    Returns
    -------
    dict with keys:
        - azimuth_match: bool (all azimuths match)
        - range_match: bool (all ranges match)
        - azimuth_mismatches: list of sweep indices with mismatches
        - range_mismatches: list of sweep indices with mismatches
    """
    try:
        vcp_config = load_json_config("vcp_nexrad.json")
        vcp_key = f"VCP-{vcp_number}"

        if vcp_key not in vcp_config:
            return {
                "azimuth_match": None,
                "range_match": None,
                "azimuth_mismatches": [],
                "range_mismatches": [],
            }

        expected_azimuth = vcp_config[vcp_key]["dims"]["azimuth"]
        expected_range = vcp_config[vcp_key]["dims"]["range"]

        # Compare lengths first
        if len(azimuth_actual) != len(expected_azimuth):
            return {
                "azimuth_match": False,
                "range_match": False,
                "azimuth_mismatches": list(range(len(azimuth_actual))),
                "range_mismatches": list(range(len(range_actual))),
            }

        # Find mismatches (handle None values)
        azimuth_mismatches = [
            i
            for i in range(len(azimuth_actual))
            if azimuth_actual[i] is not None
            and azimuth_actual[i] != expected_azimuth[i]
        ]
        range_mismatches = [
            i
            for i in range(len(range_actual))
            if range_actual[i] is not None and range_actual[i] != expected_range[i]
        ]

        return {
            "azimuth_match": len(azimuth_mismatches) == 0,
            "range_match": len(range_mismatches) == 0,
            "azimuth_mismatches": azimuth_mismatches,
            "range_mismatches": range_mismatches,
        }
    except Exception:
        return {
            "azimuth_match": None,
            "range_match": None,
            "azimuth_mismatches": [],
            "range_mismatches": [],
        }


def collapse_split_cuts(elevations: list[float]) -> list[float]:
    """
    Remove consecutive duplicate elevations (split cuts) to get logical tilts.

    NEXRAD uses split-cut scanning where each elevation appears in pairs
    (high PRF + low PRF for velocity dealiasing). This function removes
    consecutive duplicates to reveal the logical tilt sequence.

    Example:
        [0.5, 0.5, 0.9, 0.9, 1.3, 1.3] → [0.5, 0.9, 1.3]

    Parameters
    ----------
    elevations : List[float]
        Elevation angles (rounded to 0.1°)

    Returns
    -------
    List[float]
        Collapsed elevation sequence with consecutive duplicates removed
    """
    if not elevations:
        return []

    collapsed = [elevations[0]]
    for i in range(1, len(elevations)):
        # Use tolerance for floating point comparison
        if abs(elevations[i] - elevations[i - 1]) > 0.05:
            collapsed.append(elevations[i])

    return collapsed


def classify_dynamic_type(
    elevation_angles: list[float | None],
    act_sweeps: int,
    exp_sweeps: int | None,
) -> dict[str, Any]:
    """
    Classify NEXRAD dynamic scan modes using collapsed elevation logic.

    Steps:
    1. Round elevations to nearest 0.1° (handle 0.48→0.5, 0.88→0.9)
    2. Collapse split cuts (remove consecutive duplicates)
    3. On collapsed list:
       - Count 0.5° occurrences for SAILS detection
       - Look for non-adjacent low-level block repeats for MRLE detection
       - Compare to expected elevations for AVSET detection

    Detection Rules:
    - SAILS/MESO-SAILS: Count 0.5° in collapsed list
      * 1 occurrence = base only (valid)
      * 2 occurrences = SAILS×1 (base + 1 supplemental)
      * 3 occurrences = MESO-SAILS×2 (base + 2 supplemental)
      * 4 occurrences = MESO-SAILS×3 (base + 3 supplemental)

    - MRLE: Low-level block repeats non-adjacently in collapsed list
      * [0.5, 0.9] → MRLE×2
      * [0.5, 0.9, 1.3] → MRLE×3
      * [0.5, 0.9, 1.3, 1.8] → MRLE×4

    - AVSET: Collapsed list ends early or has fewer tilts than expected
      * Detected via act_sweeps < exp_sweeps

    - MRLE takes precedence over SAILS if both patterns detected

    Parameters
    ----------
    elevation_angles : List[Optional[float]]
        Raw elevation angles from NEXRAD file
    act_sweeps : int
        Actual number of sweeps in volume
    exp_sweeps : Optional[int]
        Expected number of sweeps for this VCP

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - dynamic_type: Classification ("SAILS", "MESO-SAILS×N", "MRLE×N",
                       "AVSET", combinations, or "valid")
        - sails_inserts: Number of supplemental 0.5° scans (0-3)
        - mrle_level: MRLE level (0, 2, 3, or 4)
        - has_avset: Boolean indicating early termination
        - collapsed_elevations: List of logical tilts (split cuts removed)
        - elevation_pattern: Human-readable description
    """
    if not elevation_angles or all(e is None for e in elevation_angles):
        return {
            "dynamic_type": None,
            "sails_inserts": 0,
            "mrle_level": 0,
            "has_avset": False,
            "collapsed_elevations": None,
            "elevation_pattern": "invalid",
        }

    # Round to nearest 0.1° and filter out None values
    rounded = [round(e, 1) for e in elevation_angles if e is not None]

    if len(rounded) < 2:
        return {
            "dynamic_type": None,
            "sails_inserts": 0,
            "mrle_level": 0,
            "has_avset": False,
            "collapsed_elevations": None,
            "elevation_pattern": "insufficient_data",
        }

    # Collapse split cuts to get logical tilt sequence
    collapsed = collapse_split_cuts(rounded)

    # Check for AVSET (early termination)
    has_avset = exp_sweeps is not None and act_sweeps < exp_sweeps

    # Detect MRLE first (takes precedence over SAILS)
    mrle_level = 0
    mrle_blocks = {
        4: [0.5, 0.9, 1.3, 1.8],
        3: [0.5, 0.9, 1.3],
        2: [0.5, 0.9],
    }

    for level in [4, 3, 2]:
        block = mrle_blocks[level]
        block_len = len(block)

        # Check if block appears at start of collapsed list
        if len(collapsed) >= block_len and collapsed[:block_len] == block:
            # Look for non-adjacent repeat (skip at least 1 tilt past initial block)
            search_start = block_len + 1

            for i in range(search_start, len(collapsed) - block_len + 1):
                if collapsed[i : i + block_len] == block:
                    mrle_level = level
                    break

        if mrle_level > 0:
            break

    # If MRLE found, classification is MRLE (not SAILS)
    if mrle_level > 0:
        dynamic_type = f"MRLE×{mrle_level}"
        if has_avset:
            dynamic_type += "+AVSET"
        pattern = (
            f"MRLE×{mrle_level} ({mrle_blocks[mrle_level]} block rescanned mid-volume)"
        )
        sails_inserts = 0

    else:
        # Check for SAILS (count 0.5° in collapsed list)
        count_05 = collapsed.count(0.5)
        sails_inserts = max(0, count_05 - 1)  # Subtract base occurrence

        if sails_inserts == 0:
            # No SAILS inserts
            if has_avset:
                dynamic_type = "AVSET"
                pattern = f"Early termination ({act_sweeps}/{exp_sweeps} sweeps)"
            else:
                dynamic_type = "valid"
                pattern = "Standard VCP"

        elif sails_inserts == 1:
            dynamic_type = "SAILS"
            if has_avset:
                dynamic_type += "+AVSET"
            pattern = "SAILS×1 (base + 1 supplemental 0.5° scan)"

        else:  # 2 or 3 inserts
            dynamic_type = f"MESO-SAILS×{sails_inserts}"
            if has_avset:
                dynamic_type += "+AVSET"
            pattern = f"MESO-SAILS×{sails_inserts} (base + {sails_inserts} supplemental 0.5° scans)"

    return {
        "dynamic_type": dynamic_type,
        "sails_inserts": sails_inserts,
        "mrle_level": mrle_level,
        "has_avset": has_avset,
        "collapsed_elevations": collapsed,
        "elevation_pattern": pattern,
    }


def extract_vcp_and_shapes(file_path: str) -> dict[str, Any]:
    """
    Extract comprehensive metadata from a NEXRAD Level 2 file.

    Returns a dict with per-volume metadata and per-sweep shapes including:
    - VCP number and sweep counts
    - Elevation angles and timing for each sweep
    - Dynamic scan classification (SAILS, MRLE, AVSET)
    - File status (valid, dynamic, corrupted)
    - Per-sweep azimuth counts and moment dimensions

    Parameters
    ----------
    file_path : str
        Path to NEXRAD Level 2 file (local or S3)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive metadata with keys:
        - filename, site, timestamp, date
        - vcp, vcp_number, exp_sweeps, act_sweeps
        - dynamic, file_status
        - elevation_angles, sweep_start_times, sweep_end_times
        - dynamic_type, sails_inserts, mrle_level, elevation_pattern
        - azimuth_per_sweep, range_ref_per_sweep
        - azimuth_match, range_match, azimuth_mismatches, range_mismatches
        - (optional) shape_signature, sweep_shapes_json
    """
    try:
        _file = normalize_input_for_xradar(file_path)
        shape: dict[str, Any] = {}
        with NEXRADLevel2File(_file, loaddata=False) as nex:
            # VCP and sweep counts
            msg5 = nex.get_msg_5_data()
            vcp_number = msg5.get("pattern_number")
            vcp = f"VCP-{vcp_number}" if vcp_number is not None else None
            exp_sweeps = msg5.get("number_elevation_cuts")
            try:
                act_sweeps = len(nex.msg_31_data_header)
            except Exception:
                act_sweeps = None

            # Sweep-level shapes from headers only
            n_groups = len(getattr(nex, "msg_31_header", []))
            elevation_angles = []
            sweep_start_times = []
            sweep_end_times = []

            for group in range(n_groups):
                msg_31_header = nex.msg_31_header[group]
                sweep_key = f"sweep_{group}"
                sweep_info: dict[str, Any] = {"az": len(msg_31_header)}

                # Extract elevation angle
                try:
                    elev = round(
                        nex.msg_5["elevation_data"][group]["elevation_angle"], 2
                    )
                    elevation_angles.append(elev)
                except Exception:
                    elevation_angles.append(None)

                # Extract sweep start time (first azimuth)
                try:
                    date_start = (msg_31_header[0]["collect_date"] - 1) * 86400e3
                    milli_start = msg_31_header[0]["collect_ms"]
                    time_ms_start = date_start + milli_start
                    start_ts = to_datetime(time_ms_start, unit="ms", utc=True)
                    sweep_start_times.append(start_ts.isoformat())
                except Exception:
                    sweep_start_times.append(None)

                # Extract sweep end time (last azimuth)
                try:
                    date_end = (msg_31_header[-1]["collect_date"] - 1) * 86400e3
                    milli_end = msg_31_header[-1]["collect_ms"]
                    time_ms_end = date_end + milli_end
                    end_ts = to_datetime(time_ms_end, unit="ms", utc=True)
                    sweep_end_times.append(end_ts.isoformat())
                except Exception:
                    sweep_end_times.append(None)

                try:
                    data_hdrs = nex.data[group]["msg_31_data_header"]
                    for key in data_hdrs.keys():
                        if key not in ["VOL", "ELV", "RAD"]:
                            ng = data_hdrs[key].get("ngates")
                            if ng is not None:
                                sweep_info[f"{key}/ngates"] = ng
                except Exception:
                    pass
                shape[sweep_key] = sweep_info

        # Per-sweep arrays
        azimuth_per_sweep: list[int | None] = []
        range_ref_per_sweep: list[int | None] = []
        for i in range(len(shape)):
            s = shape.get(f"sweep_{i}", {})
            azimuth_per_sweep.append(int(s.get("az", 0)) if "az" in s else None)
            rng = s.get("REF/ngates")
            range_ref_per_sweep.append(
                int(rng) if isinstance(rng, (int, float)) else None
            )

        # Count non-None azimuth values
        azimuth_sweep_count = len([x for x in azimuth_per_sweep if x is not None])

        # Parse site and timestamp
        site = _parse_site_from_path(file_path)
        ts = extract_timestamp(file_path)
        date = ts.date().isoformat() if ts is not None else None
        dynamic = (
            (exp_sweeps != act_sweeps)
            if (exp_sweeps is not None and act_sweeps is not None)
            else None
        )
        signature = _compute_signature(shape)

        # Determine file status: corrupted vs dynamic vs valid
        if act_sweeps is not None and azimuth_sweep_count != act_sweeps:
            file_status = "corrupted"
        elif (
            act_sweeps is not None
            and exp_sweeps is not None
            and act_sweeps < exp_sweeps
        ):
            file_status = "dynamic"
        else:
            file_status = "valid"

        # Compare with VCP configuration
        comparison = (
            compare_with_vcp_config(
                azimuth_per_sweep, range_ref_per_sweep, int(vcp_number)
            )
            if vcp_number is not None
            else {
                "azimuth_match": None,
                "range_match": None,
                "azimuth_mismatches": [],
                "range_mismatches": [],
            }
        )

        # Classify dynamic scan type (SAILS, MRLE, AVSET)
        classification = classify_dynamic_type(
            elevation_angles=elevation_angles,
            act_sweeps=act_sweeps,
            exp_sweeps=exp_sweeps,
        )

        # Build result dictionary with required fields
        result = {
            "filename": file_path,
            "site": site,
            "timestamp": ts,
            "date": date,
            "vcp": vcp,
            "vcp_number": int(vcp_number) if vcp_number is not None else None,
            "exp_sweeps": int(exp_sweeps) if exp_sweeps is not None else None,
            "act_sweeps": int(act_sweeps) if act_sweeps is not None else None,
            "dynamic": bool(dynamic) if dynamic is not None else None,
            "file_status": file_status,
            "elevation_angles": json.dumps(elevation_angles),
            "sweep_start_times": json.dumps(sweep_start_times),
            "sweep_end_times": json.dumps(sweep_end_times),
            "dynamic_type": classification["dynamic_type"],
            "sails_inserts": classification["sails_inserts"],
            "mrle_level": classification["mrle_level"],
            "elevation_pattern": classification["elevation_pattern"],
            "azimuth_per_sweep": json.dumps(azimuth_per_sweep),
            "range_ref_per_sweep": json.dumps(range_ref_per_sweep),
            "azimuth_match": comparison["azimuth_match"],
            "range_match": comparison["range_match"],
            "azimuth_mismatches": json.dumps(comparison["azimuth_mismatches"]),
            "range_mismatches": json.dumps(comparison["range_mismatches"]),
        }

        # Add optional fields based on config
        if INCLUDE_SHAPE_SIGNATURE:
            result["shape_signature"] = signature

        if INCLUDE_SWEEP_SHAPES_JSON:
            result["sweep_shapes_json"] = json.dumps(shape, separators=(",", ":"))

        return result
    except Exception:
        site = _parse_site_from_path(file_path)
        try:
            ts = extract_timestamp(file_path)
            date = ts.date().isoformat()
        except Exception:
            ts = None
            date = None

        # Build error result with required fields
        error_result = {
            "filename": file_path,
            "site": site,
            "timestamp": ts,
            "date": date,
            "vcp": None,
            "vcp_number": None,
            "exp_sweeps": None,
            "act_sweeps": None,
            "dynamic": None,
            "file_status": None,
            "elevation_angles": json.dumps([]),
            "sweep_start_times": json.dumps([]),
            "sweep_end_times": json.dumps([]),
            "dynamic_type": None,
            "sails_inserts": 0,
            "mrle_level": 0,
            "elevation_pattern": "error",
            "azimuth_per_sweep": json.dumps([]),
            "range_ref_per_sweep": json.dumps([]),
            "azimuth_match": None,
            "range_match": None,
            "azimuth_mismatches": json.dumps([]),
            "range_mismatches": json.dumps([]),
        }

        # Add optional fields based on config
        if INCLUDE_SHAPE_SIGNATURE:
            error_result["shape_signature"] = ""

        if INCLUDE_SWEEP_SHAPES_JSON:
            error_result["sweep_shapes_json"] = json.dumps({})

        return error_result


def process(
    files: Sequence[str], output_csv: str | None = None
) -> list[dict[str, Any]]:
    """
    Process radar files with optional incremental CSV writing.

    For large datasets (>10K files), writes results to CSV incrementally
    to avoid memory issues. For smaller datasets, gathers all results first.

    Parameters
    ----------
    files : Sequence[str]
        List of file paths to process
    output_csv : Optional[str]
        Path to output CSV file (optional)

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries, one per file
    """
    if USE_DASK and DASK_AVAILABLE:
        cluster = LocalCluster(
            dashboard_address=f":{DASHBOARD_PORT}",
        )
        client = Client(cluster)

        # Use client.map() for efficient graph construction
        futures = client.map(extract_vcp_and_shapes, files)

        # For large datasets: write incrementally to CSV
        if output_csv and INCREMENTAL_CSV and len(files) > 10000:
            import os

            results = []
            file_exists = os.path.exists(output_csv)

            with open(output_csv, "a", newline="") as f:
                # Build fieldnames list dynamically based on config
                fieldnames = [
                    "filename",
                    "site",
                    "timestamp",
                    "date",
                    "vcp",
                    "vcp_number",
                    "exp_sweeps",
                    "act_sweeps",
                    "dynamic",
                    "file_status",
                    "elevation_angles",
                    "sweep_start_times",
                    "sweep_end_times",
                    "dynamic_type",
                    "sails_inserts",
                    "mrle_level",
                    "elevation_pattern",
                    "azimuth_per_sweep",
                    "range_ref_per_sweep",
                    "azimuth_match",
                    "range_match",
                    "azimuth_mismatches",
                    "range_mismatches",
                ]

                # Add optional fields if enabled
                if INCLUDE_SHAPE_SIGNATURE:
                    fieldnames.append("shape_signature")
                if INCLUDE_SWEEP_SHAPES_JSON:
                    fieldnames.append("sweep_shapes_json")

                # Add year/month at end
                fieldnames.extend(["year", "month"])

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                # Process results as they complete
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)

                    # Add year/month for CSV
                    if result.get("timestamp"):
                        dt = pd.to_datetime(result["timestamp"])
                        result["year"] = dt.year
                        result["month"] = dt.month

                    writer.writerow(result)

                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i + 1}/{len(files)} files")
                        f.flush()

            print(f"✓ Incremental CSV written to {output_csv}")
            return results
        else:
            # For smaller datasets: gather all at once
            results = client.gather(futures)
            return results

        # No explicit shutdown - avoids heartbeat errors

    else:
        return [extract_vcp_and_shapes(f) for f in files]


def write_parquet(rows: list[dict[str, Any]], root: str) -> str | None:
    """
    Write metadata results to Parquet format.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Metadata dictionaries to write
    root : str
        Root directory for output

    Returns
    -------
    Optional[str]
        Path to written Parquet file, or None if failed
    """
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Add year/month columns for filtering (not partitioning)
    if "timestamp" in df.columns:
        df["year"] = pd.to_datetime(df["timestamp"]).dt.year
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    else:
        df["year"] = None
        df["month"] = None
    # ensure column order - build dynamically based on config
    preferred = [
        "filename",
        "site",
        "timestamp",
        "date",
        "vcp",
        "vcp_number",
        "exp_sweeps",
        "act_sweeps",
        "dynamic",
        "file_status",
        "elevation_angles",
        "sweep_start_times",
        "sweep_end_times",
        "dynamic_type",
        "sails_inserts",
        "mrle_level",
        "elevation_pattern",
        "azimuth_per_sweep",
        "range_ref_per_sweep",
        "azimuth_match",
        "range_match",
        "azimuth_mismatches",
        "range_mismatches",
    ]

    # Add optional fields if enabled
    if INCLUDE_SHAPE_SIGNATURE:
        preferred.append("shape_signature")
    if INCLUDE_SWEEP_SHAPES_JSON:
        preferred.append("sweep_shapes_json")

    # Add year/month at end
    preferred.extend(["year", "month"])

    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    df = df[cols]

    try:
        import os

        os.makedirs(root, exist_ok=True)

        # Write single parquet file (no partitioning)
        output_file = os.path.join(root, "nexrad_metadata.parquet")
        df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)
        print(f"Wrote {len(df)} rows to {output_file}")
        return output_file
    except Exception as e:
        print(f"Parquet write failed ({e}); skipping Parquet output.")
        return None


def main(files: list[str] | None = None) -> int:
    """
    Main entry point for metadata extraction.

    Parameters
    ----------
    files : Optional[List[str]]
        List of file paths to process. If None, uses get_radar_files_async()
        to fetch files based on hardcoded query (edit this for your use case).

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)

    Examples
    --------
    >>> # Run with custom file list
    >>> from raw2zarr.utils.metadata_extractor import main
    >>> files = ["s3://path/to/file1", "s3://path/to/file2"]
    >>> main(files=files)

    >>> # Run from command line with default settings
    >>> python -m raw2zarr.utils.metadata_extractor
    """
    if files is None:
        # Example: Fetch files using get_radar_files_async
        # Edit this section for your use case
        import asyncio
        from datetime import datetime

        from ..utils import get_radar_files_async

        try:
            files = asyncio.run(
                get_radar_files_async(
                    radar_site="KLOT",
                    start_time=datetime(2025, 3, 14, 0, 0),
                    end_time=datetime(2025, 3, 15, 23, 59),
                )
            )
        except Exception as e:
            print(f"Error fetching files: {e}")
            return 1

    if not files:
        print("No files found. Provide files as argument to main().")
        return 1

    print(f"Processing {len(files)} radar files...")

    # Process files (with optional incremental CSV writing for large datasets)
    if len(files) > 10000 and INCREMENTAL_CSV and OUTPUT_CSV:
        # Incremental CSV mode (memory efficient for large datasets)
        results = process(files, output_csv=OUTPUT_CSV)
    else:
        # Standard mode: gather all results first
        results = process(files)

        # Write CSV for smaller datasets (not already written incrementally)
        if OUTPUT_CSV and (not INCREMENTAL_CSV or len(files) <= 10000):
            try:
                with open(OUTPUT_CSV, "w", newline="") as fh:
                    # Build fieldnames list dynamically based on config
                    fieldnames = [
                        "filename",
                        "site",
                        "timestamp",
                        "date",
                        "vcp",
                        "vcp_number",
                        "exp_sweeps",
                        "act_sweeps",
                        "dynamic",
                        "file_status",
                        "elevation_angles",
                        "sweep_start_times",
                        "sweep_end_times",
                        "dynamic_type",
                        "sails_inserts",
                        "mrle_level",
                        "elevation_pattern",
                        "azimuth_per_sweep",
                        "range_ref_per_sweep",
                        "azimuth_match",
                        "range_match",
                        "azimuth_mismatches",
                        "range_mismatches",
                    ]

                    # Add optional fields if enabled
                    if INCLUDE_SHAPE_SIGNATURE:
                        fieldnames.append("shape_signature")
                    if INCLUDE_SWEEP_SHAPES_JSON:
                        fieldnames.append("sweep_shapes_json")

                    # Add year/month at end
                    fieldnames.extend(["year", "month"])

                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in results:
                        # Add year/month if not present
                        if row.get("timestamp") and "year" not in row:
                            dt = pd.to_datetime(row["timestamp"])
                            row["year"] = dt.year
                            row["month"] = dt.month
                        writer.writerow(row)

                print(f"✓ CSV written to {OUTPUT_CSV}")
            except Exception as e:
                print(f"Failed to write CSV '{OUTPUT_CSV}': {e}")

    # Write parquet (skip for very large datasets to avoid memory issues)
    if WRITE_PARQUET and len(files) < 50000:
        out = write_parquet(results, PARQUET_ROOT)
        if out:
            print(f"✓ Parquet written to {out}")
    elif WRITE_PARQUET and len(files) >= 50000:
        print(
            f"⚠ Skipping parquet for {len(files)} files (too large for in-memory write)"
        )
        print("  Consider converting CSV → Parquet in batches later")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
