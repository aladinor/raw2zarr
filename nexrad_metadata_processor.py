#!/usr/bin/env python3
"""
NEXRAD Metadata Processor

Lightweight utility to extract VCP numbers (and basic metadata in the future)
from NEXRAD Level 2 files. Designed to run locally or distributed via Dask.

For now, the primary output is a list of tuples: (filename, vcp_string),
where vcp_string looks like "VCP-12".

Example (programmatic):
    from nexrad_metadata_processor import process_nexrad_files
    results = process_nexrad_files(["s3://.../KVNX20110520_000023_V06"], n_workers=4)

Example (CLI):
    python nexrad_metadata_processor.py --engine nexradlevel2 --max-files 10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import os
from pathlib import Path

from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

from raw2zarr.io.preprocess import normalize_input_for_xradar


# Optional Dask support
try:
    from dask.distributed import Client, LocalCluster, as_completed

    DASK_AVAILABLE = True
except Exception:
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    as_completed = None  # type: ignore
    DASK_AVAILABLE = False


def extract_nexrad_vcp(file_path: str) -> Tuple[str, Optional[str]]:
    """Extract VCP string (e.g., "VCP-12") from a single NEXRAD Level 2 file.

    Parameters
    ----------
    file_path : str
        Path or URL to a NEXRAD Level 2 file. Supports local, S3, and other fsspec-URIs
        handled by `normalize_input_for_xradar`.

    Returns
    -------
    tuple
        (filename, vcp) where vcp is a string like "VCP-12" or None if detection failed.
    """
    try:
        _file = normalize_input_for_xradar(file_path)
        msg5 = NEXRADLevel2File(_file).get_msg_5_data()
        vcp_number = msg5.get("pattern_number")
        vcp = f"VCP-{vcp_number}" if vcp_number is not None else None
        return file_path, vcp
    except Exception:
        return file_path, None


def process_nexrad_files(
    files: Sequence[str],
    *,
    use_dask: bool = True,
    n_workers: int = 4,
    threads_per_worker: int = 1,
) -> List[Tuple[str, Optional[str]]]:
    """Process a list of NEXRAD files and extract VCP strings.

    Parameters
    ----------
    files : Sequence[str]
        List/sequence of file paths or URLs.
    use_dask : bool, optional
        If True and Dask is available, use a local Dask cluster, by default True.
    n_workers : int, optional
        Number of Dask workers for the local cluster, by default 4.
    threads_per_worker : int, optional
        Threads per Dask worker, by default 1.

    Returns
    -------
    list[tuple[str, Optional[str]]]
        List of (filename, vcp_string or None).
    """
    if use_dask and DASK_AVAILABLE:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        try:
            futures = [client.submit(extract_nexrad_vcp, f) for f in files]
            results: List[Tuple[str, Optional[str]]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
            return results
        finally:
            client.close()
            cluster.close()
    else:
        return [extract_nexrad_vcp(f) for f in files]


def _get_sample_nexrad_files(max_files: Optional[int] = None) -> List[str]:
    """Best-effort helper to retrieve sample NEXRAD file URLs using repo utility.

    This relies on `delete.get_radar_files("nexradlevel2")` which in turn pulls from
    the Unidata S3 public bucket. Use only for quick testing.
    """
    try:
        from delete import get_radar_files

        files, _zs, engine, _cfg = get_radar_files("nexradlevel2")
        if engine != "nexradlevel2":
            return []
        return files[:max_files] if max_files else files
    except Exception:
        return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract NEXRAD VCP metadata")
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths/URLs to NEXRAD Level 2 files. If omitted, fetch sample files.",
    )
    parser.add_argument("--max-files", type=int, default=10, help="Limit for sample files")
    parser.add_argument(
        "--no-dask", dest="use_dask", action="store_false", help="Disable Dask processing"
    )
    parser.set_defaults(use_dask=True)

    args = parser.parse_args(argv)

    files = list(args.files) if args.files else _get_sample_nexrad_files(args.max_files)
    if not files:
        print("No files provided or sample retrieval failed.")
        return 1

    results = process_nexrad_files(files, use_dask=args.use_dask)

    # Print a simple CSV-like output: filename,vcp
    for fname, vcp in results:
        print(f"{fname},{vcp if vcp is not None else ''}")

    # Also emit JSON to stdout if requested via environment variable
    if os.environ.get("NEXRAD_META_JSON", "0") == "1":
        print(json.dumps([{"filename": f, "vcp": v} for f, v in results]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

