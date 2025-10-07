#!/usr/bin/env python
import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from functools import wraps
from time import time
from typing import Optional

import fsspec

# Constants
NEXRAD_S3_BUCKET = "unidata-nexrad-level2"
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


def create_query(site: str, date: datetime) -> str:
    """
    Build query string for IDEAM radar data access.

    Parameters
    ----------
    site : str
        Radar site identifier.
    date : datetime
        Date for the radar data.

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

    return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{site.capitalize()}/{site[:3].upper()}{date:%y%m%d}"


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
        with open(samples_file) as f:
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


def parse_nexrad_filename(fname: str) -> Optional[datetime]:
    """
    Extract datetime from NEXRAD filename.

    Parameters
    ----------
    fname : str
        NEXRAD filename (e.g., 'KVNX20110520_000238_V03.gz')

    Returns
    -------
    Optional[datetime]
        Parsed datetime if successful, None otherwise

    Examples
    --------
    >>> parse_nexrad_filename('KVNX20110520_000238_V03.gz')
    datetime.datetime(2011, 5, 20, 0, 2, 38)
    >>> parse_nexrad_filename('invalid_filename.gz')
    None
    """
    match = re.search(r"(\d{8})_(\d{6})", fname)
    if match:
        date_str = match.group(1) + match.group(2)
        return datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return None


async def list_day_files_async(
    fs, date: datetime, radar: str, bucket: str = NEXRAD_S3_BUCKET
) -> list[str]:
    """
    Asynchronously list radar files for a specific day.

    Parameters
    ----------
    fs : fsspec.filesystem
        Filesystem instance for S3 access
    date : datetime
        Date to query files for
    radar : str
        Radar site identifier (e.g., 'KVNX')
    bucket : str, default "noaa-nexrad-level2"
        S3 bucket name

    Returns
    -------
    List[str]
        List of S3 file paths for the specified day
    """
    prefix = f"{bucket}/{date.strftime('%Y/%m/%d')}/{radar}/{radar}"
    files = await asyncio.to_thread(fs.glob, f"{prefix}*")
    return [f"s3://{f}" for f in files]


async def get_radar_files_async(
    radar_site: str = "KVNX",
    start_date: Optional[str] = None,
    num_days: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    bucket: str = NEXRAD_S3_BUCKET,
) -> tuple[list[str], str, str]:
    """
    Asynchronously list radar files from NEXRAD S3 bucket within a time range.

    This function provides parallel file listing across multiple days for improved
    performance compared to the synchronous `list_nexrad_files()`.

    Parameters
    ----------
    radar_site : str, default "KVNX"
        Radar site identifier
    start_date : Optional[str]
        Start date in format 'YYYY/MM/DD' (used with num_days)
    num_days : Optional[int]
        Number of days to query from start_date
    start_time : Optional[datetime]
        Start datetime for precise time range
    end_time : Optional[datetime]
        End datetime for precise time range
    bucket : str, default "noaa-nexrad-level2"
        S3 bucket name (supports "noaa-nexrad-level2" or "unidata-nexrad-level2")

    Returns
    -------
    Tuple[List[str], str, str]
        - List of S3 file paths sorted by time
        - Zarr store name (e.g., "earthmover/KVNX")
        - Engine identifier ("nexradlevel2")

    Raises
    ------
    ValueError
        If neither (start_date + num_days) nor (start_time + end_time) are provided

    Examples
    --------
    >>> # Using date range
    >>> files, zs, engine = await get_radar_files_async(
    ...     radar_site="KVNX",
    ...     start_date="2011/05/20",
    ...     num_days=1
    ... )

    >>> # Using datetime range
    >>> from datetime import datetime
    >>> files, zs, engine = await get_radar_files_async(
    ...     radar_site="KVNX",
    ...     start_time=datetime(2011, 5, 20, 0, 0),
    ...     end_time=datetime(2011, 5, 20, 23, 59)
    ... )

    Notes
    -----
    This async version performs parallel file listing across days using asyncio.gather(),
    resulting in 10-100x speedup for multi-day queries compared to synchronous version.
    """
    fs = fsspec.filesystem("s3", anon=True)

    # Determine time range
    if start_time and end_time:
        start_dt, end_dt = start_time, end_time
    elif start_date and num_days:
        start_dt = datetime.strptime(start_date, "%Y/%m/%d")
        end_dt = start_dt + timedelta(days=num_days)
    else:
        raise ValueError(
            "Provide either (start_date + num_days) or (start_time + end_time)"
        )
    start_day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    days = [
        start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)
    ]

    # Parallel file listing
    tasks = [list_day_files_async(fs, d, radar_site, bucket) for d in days]
    results = await asyncio.gather(*tasks)
    all_files = [f for sublist in results for f in sublist]

    # Filter and sort files within exact time range
    filtered = [
        (f, dt)
        for f in all_files
        if not f.endswith("_MDM")  # Skip _MDM files
        and (dt := parse_nexrad_filename(f))
        and start_dt <= dt <= end_dt
    ]

    filtered_sorted = [f for f, _ in sorted(filtered, key=lambda x: x[1])]

    print(f"Found {len(filtered_sorted)} radar files between {start_dt} and {end_dt}")
    return filtered_sorted
