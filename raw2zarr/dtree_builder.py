from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import xarray as xr
from xarray import DataTree
from xarray.backends.common import _normalize_path

# Relative imports
from .dtree_io import load_radar_data
from .template_manager import ScanTemplateManager
from .utils import (
    _get_missing_elevations,
    batch,
    check_dynamic_scan,
    dtree_encoding,
    ensure_dimension,
    fix_angle,
    get_vcp_values,
    load_json_config,
)
from .zarr_writer import dtree2zarr


def datatree_builder(
    filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
    engine: str = "iris",
    append_dim: str = "vcp_time",
) -> DataTree:
    """
    Construct a hierarchical xarray.DataTree from radar data files.

    This function loads radar data from one or more files and organizes it into a nested
    `xarray.DataTree` structure. The data can be processed in batches and supports different
    backend engines for reading the data.

    Parameters:
        filename_or_obj (str | os.PathLike | Iterable[str | os.PathLike]):
            Path or paths to the radar data files to be loaded. Can be a single file,
            a directory path, or an iterable of file paths.
        engine (str, optional):
            The backend engine to use for loading the radar data. Common options include
            'iris' (default) and 'odim'. The selected engine must be supported by the underlying
            data processing libraries.
        append_dim (str, optional):
            The name of the dimension to use for concatenating data across files. Default is 'vcp_time'.
            Note: The 'time' dimension cannot be used as the concatenation dimension because it is
            already a predefined dimension in the dataset and reserved for temporal data. Choose
            a unique dimension name that does not conflict with existing dimensions in the datasets.


    Returns:
        xarray.DataTree:
            A nested `xarray.DataTree` object that combines all the loaded radar data files into a
            hierarchical structure. Each node in the tree corresponds to an `xarray.Dataset`.

    Raises:
        ValueError:
            If no files are successfully loaded or if all batches result in empty data.

    Notes:
        - This function is designed to handle large datasets efficiently, potentially
          processing data in batches and leveraging parallelism if supported by the backend.
        - The resulting `xarray.DataTree` retains a hierarchical organization based on the structure
          of the input files and their metadata.

    Example:
        >>> from raw2zarr import datatree_builder
        >>> tree = datatree_builder(["file1.RAW", "file2.RAW"], engine="iris", append_dim="vcp_time")
        >>> print(tree)
        >>> print(tree["root/child"].to_dataset())  # Access a node's dataset
    """

    filename_or_obj = _normalize_path(filename_or_obj)
    dtree = load_radar_data(filename_or_obj, engine=engine)
    task_name = dtree.attrs.get("scan_name", "default_task").strip()
    dtree = (dtree.pipe(fix_angle)).xradar.georeference()
    if (engine == "nexradlevel2") & check_dynamic_scan(dtree):
        dtree = align_dynamic_scan(dtree, append_dim=append_dim)
    else:
        dtree = dtree.pipe(ensure_dimension, append_dim)
    dtree = DataTree.from_dict({task_name: dtree})
    dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)
    return dtree


def process_file(file: str, engine: str = "nexradlevel2") -> DataTree:
    """
    Load and transform a single radar file into a DataTree object.
    """
    try:
        return datatree_builder(file, engine=engine)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


def append_sequential(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "iris",
    **kwargs,
) -> None:
    """
    Sequentially processes radar files and appends their data to a Zarr store.

    This function processes radar files one at a time, converting each file into an
    `xarray.DataTree` object and sequentially appending its data to the specified Zarr store.
    The process ensures data is written in an ordered manner along the specified dimension.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            An iterable containing file paths to the radar data files to be processed.
        append_dim (str):
            The dimension along which data is appended in the Zarr store. Typically used
            to represent temporal or scan-specific dimensions (e.g., "vcp_time").
        zarr_store (str):
            The file path or URL to the output Zarr store where data will be appended.
        engine (str, optional):
            The backend engine to use for loading radar data. Options include:
            - "iris" (default): For IRIS format radar data.
            - "nexradlevel2": For NEXRAD Level 2 data.
            - "odim": For ODIM HDF5 format.
        **kwargs:
            Additional optional parameters, including:
            - zarr_format (int, optional): The Zarr format version to use (default: 2).

    Returns:
        None:
            The function does not return any values. Processed radar data is written
            directly to the specified Zarr store.

    Raises:
        ValueError:
            If an error occurs during appending to the Zarr store or if the provided
            dimension or file paths are invalid.

    Notes:
        - Data is written sequentially to the Zarr store, ensuring an ordered structure
          along the specified `append_dim`.
        - Handles encoding for compatibility with the Zarr format, including time and
          custom dimension variables.
        - Supports customization via the `engine` parameter for different radar data formats.
    """
    for file in radar_files:
        dtree = process_file(file, engine=engine)
        if dtree:
            enc = dtree.encoding
            dtree2zarr(
                dtree,
                store=zarr_store,
                mode="a-",
                encoding=enc,
                consolidated=consolidated,
                zarr_format=zarr_format,
                append_dim=append_dim,
                write_inherited_coords=True,
            )


def append_parallel(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "nexradlevel2",
    batch_size: int = None,
    **kwargs,
) -> None:
    """
    Load radar files in parallel and append their data sequentially to a Zarr store.

    This function uses Dask Bag to load radar files in parallel, processing them in
    configurable batches. After loading, the resulting `xarray.DataTree` objects are
    processed and written sequentially to the Zarr store, ensuring consistent and ordered
    data storage. A Dask LocalCluster is used to distribute computation across available cores.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            An iterable containing paths to the radar files to process.
        append_dim (str):
            The dimension along which to append data in the Zarr store.
        zarr_store (str):
            The path to the output Zarr store where data will be written.
        engine (str, optional):
            The backend engine used to load radar files. Defaults to "nexradlevel2".
        batch_size (int, optional):
            The number of files to process in each batch. If not specified, it defaults to
            the total number of cores available in the Dask cluster.
        **kwargs:
            Additional arguments, including:
                - zarr_format (int, optional): The Zarr format version to use (default: 2).

    Returns:
        None:
            This function writes data directly to the specified Zarr store and does not return a value.

    Notes:
        - File loading is parallelized using Dask Bag for efficiency, but data writing
          to the Zarr store is performed sequentially to ensure consistent and ordered output.
        - A Dask LocalCluster is created with a web-based dashboard for monitoring at
          `http://127.0.0.1:8785` by default.
        - If `batch_size` is not specified, it is automatically set based on the available cores
          in the Dask cluster.
    """

    import gc
    from functools import partial

    from dask import bag as db
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(dashboard_address="127.0.0.1:8785", memory_limit="10GB")
    client = Client(cluster)
    pf = partial(process_file, engine=engine)

    if not batch_size:
        batch_size = sum(client.ncores().values())

    for radar_files_batch in batch(radar_files, n=batch_size):
        bag = db.from_sequence(radar_files_batch, npartitions=batch_size).map(pf)
        ls_dtree: list[DataTree] = bag.compute()
        for dtree in ls_dtree:
            if dtree:
                dtree2zarr(
                    dtree,
                    store=zarr_store,
                    mode="a-",
                    encoding=dtree.encoding,
                    consolidated=consolidated,
                    zarr_format=zarr_format,
                    append_dim=append_dim,
                    compute=True,
                    write_inherited_coords=True,
                )
        del bag, ls_dtree
        gc.collect()


def align_dynamic_scan(tree: DataTree, append_dim: str = "vcp_time") -> DataTree:
    vcp = tree.attrs["scan_name"]
    VCP_REFERENCE = get_vcp_values(vcp_name=vcp)
    actual_sweeps = [
        tree[path].ds["sweep_fixed_angle"].values.item()
        for path in tree.match("sweep_*").children
    ]
    missing_idx = _get_missing_elevations(VCP_REFERENCE, actual_sweeps)
    sweep_groups = {
        "root": [("/", tree.root.to_dataset())],
        "radar_parameters": [
            ("radar_parameters", tree["radar_parameters"].to_dataset())
        ],
        "georeferencing_correction": [
            (
                "georeferencing_correction",
                tree["georeferencing_correction"].to_dataset(),
            )
        ],
        "radar_calibration": [
            ("radar_calibration", tree["radar_calibration"].to_dataset())
        ],
    }
    for node_name, node in tree.match("sweep_*").items():
        ds = node.ds  # Access the xarray.Dataset for this node
        if "sweep_fixed_angle" not in ds:
            raise ValueError(f"'sweep_fixed_angle' not found in node {node_name}")

        # Extract identifying properties
        fixed_angle = ds["sweep_fixed_angle"].values.item()  # Assume single value
        dims = tuple(ds.sizes.items())  # Tuple of dimension names and sizes

        # Use (fixed_angle, dims) as a key to group similar sweeps
        key = (fixed_angle, dims)
        if key not in sweep_groups:
            sweep_groups[key] = []
        sweep_groups[key].append((node_name, ds))

    new_dtree = {}
    start_time = pd.to_datetime(tree.time_coverage_start.item())

    i = 0
    for key, node_ds_list in sweep_groups.items():
        group_path, datasets = zip(*node_ds_list)
        group_path = group_path[0]
        if group_path.startswith("sweep"):
            group_path = f"sweep_{i}"
            i += 1
        # # Separate node names and datasets
        # Unpack key components
        if len(datasets) > 1:
            # Concatenate datasets along the time dimension
            time_coords = [pd.to_datetime(ds.time.mean().values) for ds in datasets]
            time_coords[0] = start_time
            time_coords = [
                ts.tz_convert(None) if ts.tzinfo else ts for ts in time_coords
            ]
            # Create a DataArray
            time_coords_da = xr.DataArray(
                data=time_coords,
                dims=(append_dim,),
                name=append_dim,
                attrs={
                    "description": "Volume Coverage Pattern time since start of volume scan"
                },
            )
            concat_ds = xr.concat(datasets, dim=append_dim)
            concat_ds[append_dim] = time_coords_da

            # Set the new variable as a coordinate and expand the dimension
            concat_ds = concat_ds.set_coords(append_dim)
            new_dtree[group_path] = concat_ds
        else:
            # Single dataset, keep as is
            ds = datasets[0].copy()
            ds[append_dim] = start_time
            # Define attributes for the new dimension
            attrs = {
                "description": "Volume Coverage Pattern time since start of volume scan",
            }
            ds[append_dim].attrs = attrs
            # Set the new variable as a coordinate and expand the dimension
            ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)
            coords_to_expand = ["time", "elevation", "x", "y", "z"]
            # Expand each coordinate using a for loop
            for coord in coords_to_expand:
                if coord in ds.coords:  # Ensure the coordinate exists in the dataset
                    ds[coord] = ds[coord].expand_dims(dim="vcp_time", axis=0)
            new_dtree[group_path] = ds

    if missing_idx:
        template_mgr = ScanTemplateManager()
        # Get radar metadata once
        radar_info = {
            "lon": tree.root.longitude.item(),  # Changed from radar_lon
            "lat": tree.root.latitude.item(),  # Changed from radar_lat
            "alt": tree.root.altitude.item(),  # Changed from radar_alt
            "reference_time": pd.to_datetime(
                tree.time_coverage_start.item()
            ).to_numpy(),
            "vcp": tree.attrs["scan_name"],
        }
        vcp_config = load_json_config("vcp.json")[vcp]
        for idx in missing_idx:
            scan_type = vcp_config["scan_types"][idx]

            empty_ds = template_mgr.create_scan_dataset(
                scan_type=scan_type, sweep_idx=idx, radar_info=radar_info
            )

            # Use consistent naming convention
            group_path = f"sweep_{idx}"
            empty_ds[append_dim] = start_time
            # Define attributes for the new dimension
            attrs = {
                "description": "Volume Coverage Pattern time since start of volume scan",
            }
            empty_ds[append_dim].attrs = attrs
            # Set the new variable as a coordinate and expand the dimension
            new_dtree[group_path] = empty_ds.set_coords(append_dim).expand_dims(
                dim=append_dim, axis=0
            )
    radar_info = {
        "lon": tree.root.longitude.item(),
        "lat": tree.root.latitude.item(),
        "alt": tree.root.altitude.item(),
        "reference_time": pd.to_datetime(tree.time_coverage_start.item()).to_numpy(),
        "vcp_id": vcp,
    }
    new_dtree = _dtree_aligment(new_dtree, append_dim=append_dim, radar_info=radar_info)
    # Step 3: Build a new datatree with reordered nodes
    return DataTree.from_dict(new_dtree)


def _dtree_aligment(tree_dict: dict, append_dim: str, radar_info: dict) -> dict:
    vcp_id = radar_info.pop("vcp_id")
    max_vcp_time = 0
    for path, ds in tree_dict.items():
        if append_dim in ds.coords:
            len_vcp_time = len(ds.coords[append_dim].values)
            if len_vcp_time > max_vcp_time:
                max_vcp_time = len_vcp_time

    all_vcp_time_coords = set()
    for ds in tree_dict.values():
        if append_dim in ds.coords:
            all_vcp_time_coords.update(ds.coords[append_dim].values)

    # Convert the superset to a sorted array (ensure it's datetime64[ns])
    unified_time = pd.to_datetime(list(all_vcp_time_coords)[:max_vcp_time])
    for time in unified_time:
        radar_info["vcp_time"] = time
        create_empty_vcp_datatree(vcp_id=vcp_id, radar_info=radar_info)
    # Align all datasets to the unified vcp_time coordinates
    aligned_tree_dict = {}
    for path, ds in tree_dict.items():
        if append_dim in ds.coords:
            if path == "/":
                method = "pad"
            else:
                method = None
            # Reindex the dataset to include the unified coordinates, using NaT for missing values
            aligned_ds = ds.reindex({append_dim: unified_time}, method=method)
            aligned_tree_dict[path] = aligned_ds
        else:
            # If vcp_time_dim is missing, create a dataset with only the unified coordinates
            aligned_ds = xr.Dataset(coords={append_dim: unified_time})
            aligned_tree_dict[path] = aligned_ds

    return aligned_tree_dict


def create_empty_vcp_datatree(vcp_id: str, radar_info: dict) -> DataTree:
    """
    Create a DataTree with empty datasets for all scans in a VCP

    Parameters:
        vcp_id: Volume Coverage Pattern ID (e.g., "VCP-21")
        radar_info: Dictionary with radar metadata:
            - lon: Radar longitude
            - lat: Radar latitude
            - alt: Radar altitude
            - reference_time: Volume start time
            - vcp_time: VCP timestamp

    Returns:
        DataTree: Hierarchical structure with empty scans for all expected elevations
    """
    # Load VCP configuration
    vcp_config = load_json_config("vcp.json")[vcp_id]
    template_mgr = ScanTemplateManager()

    # Create empty datasets for all scans in VCP
    empty_datasets = {}
    for idx, (elevation, scan_type) in enumerate(
        zip(vcp_config["elevations"], vcp_config["scan_types"])
    ):
        empty_ds = template_mgr.create_scan_dataset(
            scan_type=scan_type,
            elevation=elevation,
            radar_info={
                **radar_info,
                "vcp_time": radar_info["vcp_time"],  # Add VCP timestamp
            },
        )

        # Use consistent naming convention
        node_name = f"sweep_{idx}"
        empty_datasets[node_name] = empty_ds

    # Create DataTree from dictionary
    return DataTree.from_dict(empty_datasets)
