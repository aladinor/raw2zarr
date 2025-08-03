#!/usr/bin/env python
import os
import json
import re
from datetime import datetime, timedelta
from functools import wraps
from time import time
from typing import Any

import fsspec

# Constants
NEXRAD_S3_BUCKET = "noaa-nexrad-level2"
NEXRAD_FILENAME_PATTERN = r"{radar}(\d{{8}})_(\d{{6}})_V\d{{2}}(?:\.gz)?$"


def timer_func(func):
    """
    Decorator that times the execution of a function.

    This decorator wraps a function and prints the execution time when the function completes.

    Parameters
    ----------
    func : callable
        The function to be timed.

    Returns
    -------
    callable
        The wrapped function that includes timing functionality.

    Examples
    --------
    >>> @timer_func
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    >>> slow_function()  # doctest: +SKIP
    Elapsed time: 1.00 seconds
    'done'
    """

    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def make_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        The directory path to create.

    Notes
    -----
    This function uses os.makedirs with exist_ok=True for safe directory creation.
    """
    os.makedirs(path, exist_ok=True)


def create_query(site: str, date: datetime, prod: str = "sigmet") -> str:
    """
    Build query string for IDEAM radar data access.

    Parameters
    ----------
    site : str
        Radar site identifier.
    date : datetime
        Date for the radar data.
    prod : str, default "sigmet"
        Product type identifier (currently not used in path generation).

    Returns
    -------
    str
        Query string for data access.

    Raises
    ------
    ValueError
        If site is empty or not a string.
    TypeError
        If date is not a datetime object.
    """
    if not site or not isinstance(site, str):
        raise ValueError("site must be a non-empty string")
    if not isinstance(date, datetime):
        raise TypeError("date must be a datetime object")

    radar_site = site.upper()
    return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"


def load_vcp_samples(samples_file: str = "data/vcp_samples.json") -> dict:
    """
    Load VCP sample data from JSON file.

    Parameters
    ----------
    samples_file : str, default "data/vcp_samples.json"
        Path to the VCP samples JSON file.

    Returns
    -------
    dict
        Dictionary containing VCP sample data with VCP names as keys and
        lists of S3 file paths as values.

    Raises
    ------
    FileNotFoundError
        If the samples file doesn't exist.
    json.JSONDecodeError
        If the file contains invalid JSON.

    Examples
    --------
    >>> samples = load_vcp_samples("data/vcp_samples.json")  # doctest: +SKIP
    >>> list(samples.keys())[:3]  # doctest: +SKIP
    ['VCP-11', 'VCP-12', 'VCP-21']
    """
    try:
        with open(samples_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"VCP samples file not found: {samples_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Error parsing VCP samples JSON: {e}", e.doc, e.pos
        ) from e


def _parse_nexrad_filename(filename: str, radar: str) -> datetime | None:
    """
    Extract datetime from NEXRAD filename.

    Parameters
    ----------
    filename : str
        NEXRAD filename to parse.
    radar : str
        4-character radar site identifier.

    Returns
    -------
    datetime | None
        Parsed datetime from filename, or None if parsing fails.
    """
    pattern = re.compile(NEXRAD_FILENAME_PATTERN.format(radar=radar))
    match = pattern.match(filename)
    if not match:
        return None

    try:
        date_part, time_part = match.groups()
        return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _get_files_for_date(
    fs: fsspec.AbstractFileSystem,
    radar: str,
    date: datetime,
    start_dt: datetime,
    end_dt: datetime,
) -> list[str]:
    """
    Get NEXRAD files for a specific date within time range.

    Parameters
    ----------
    fs : fsspec.AbstractFileSystem
        Filesystem interface for S3 access.
    radar : str
        4-character radar site identifier.
    date : datetime
        Date to search for files.
    start_dt : datetime
        Start time filter.
    end_dt : datetime
        End time filter.

    Returns
    -------
    list[str]
        List of S3 file paths for the date within time range.
    """
    date_str = date.strftime("%Y/%m/%d")
    prefix = f"{NEXRAD_S3_BUCKET}/{date_str}/{radar}/"
    file_list = []

    try:
        paths = fs.glob(f"{prefix}{radar}*")
        for path in paths:
            filename = path.split("/")[-1]
            file_datetime = _parse_nexrad_filename(filename, radar)

            if file_datetime and start_dt <= file_datetime <= end_dt:
                file_list.append(f"s3://{path}")

    except (FileNotFoundError, PermissionError, Exception):
        # Skip errors and continue processing other dates
        pass

    return file_list


def list_nexrad_files(
    radar: str = "KVNX",
    start_time: str = "2011-05-20 00:00",
    end_time: str = "2011-05-20 23:59",
) -> list[str]:
    """
    List NEXRAD Level II files from AWS S3 bucket for a given radar and time range.

    Parameters
    ----------
    radar : str, default "KVNX"
        4-character radar site identifier (e.g., "KVNX", "KTLX", "KILX").
    start_time : str, default "2011-05-20 00:00"
        Start time in format "YYYY-MM-DD HH:MM".
    end_time : str, default "2011-05-20 23:59"
        End time in format "YYYY-MM-DD HH:MM".

    Returns
    -------
    list[str]
        List of S3 paths to NEXRAD Level II files within the specified time range,
        sorted chronologically.

    Raises
    ------
    ValueError
        If time format is invalid, start_time is after end_time, or radar is invalid.

    Examples
    --------
    >>> # Get files for Moore tornado outbreak
    >>> files = list_nexrad_files("KTLX", "2013-05-20 19:00", "2013-05-20 23:00")  # doctest: +SKIP
    >>> len(files)  # doctest: +SKIP
    48

    >>> # Get single day of data
    >>> files = list_nexrad_files("KVNX", "2011-05-20 00:00", "2011-05-20 23:59")  # doctest: +SKIP

    Notes
    -----
    This function accesses the NOAA NEXRAD Level II S3 bucket at:
    s3://noaa-nexrad-level2/YYYY/MM/DD/RADAR/RADAR...

    NEXRAD filenames follow the pattern: RADARYYYYMMDD_HHMMSS_V06
    where RADAR is the 4-character site identifier.
    """
    # Input validation
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    except ValueError as e:
        raise ValueError(f"Invalid time format. Use 'YYYY-MM-DD HH:MM'. Error: {e}")

    if start_dt > end_dt:
        raise ValueError("start_time must be before or equal to end_time")

    if not radar or len(radar) != 4:
        raise ValueError("radar must be a 4-character string (e.g., 'KVNX')")

    # Initialize filesystem
    fs = fsspec.filesystem("s3", anon=True)
    file_list = []

    # Process each day in the range
    current_dt = start_dt.replace(hour=0, minute=0, second=0)
    while current_dt.date() <= end_dt.date():
        daily_files = _get_files_for_date(fs, radar, current_dt, start_dt, end_dt)
        file_list.extend(daily_files)
        current_dt += timedelta(days=1)

    return sorted(file_list)
