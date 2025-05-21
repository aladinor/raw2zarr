import os
from collections.abc import Iterable

from xarray import DataTree
from xarray.backends.common import _normalize_path

from ..io.load import load_radar_data
from ..transform.alignment import align_dynamic_scan, check_dynamic_scan, fix_angle
from ..transform.dimension import ensure_dimension
from ..transform.encoding import dtree_encoding
from ..transform.georeferencing import apply_georeferencing


def radar_datatree(
    filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
    engine: str = "iris",
    append_dim: str = "vcp_time",
) -> DataTree:
    """
    Load and transform radar file(s) into a hierarchical xarray.DataTree.

    This function reads one or more radar files and constructs a DataTree,
    applying angle correction, georeferencing, and dimension alignment.
    Adaptive scanning modes (e.g. SAILS/MRLE) are handled automatically.

    Parameters:
        filename_or_obj (str | os.PathLike | Iterable[str | os.PathLike]):
            A single file path or an iterable of radar file paths.
        engine (str, optional):
            Radar decoding backend. Supported values:
            - "iris" (default)
            - "nexradlevel2"
            - "odim"
        append_dim (str, optional):
            Name of the new dimension to append data along (default is "vcp_time").
            Cannot be an existing coordinate like "time".

    Returns:
        DataTree:
            A hierarchical container of radar datasets, where each node represents
            a processed radar sweep or metadata component.

    Raises:
        ValueError: If the scan name is missing or file loading fails.

    Example:
        >>> tree = radar_datatree("file.RAW", engine="iris")
        >>> tree["sweep_0"].to_dataset()
    """

    filename_or_obj = _normalize_path(filename_or_obj)
    dtree = load_radar_data(filename_or_obj, engine=engine)

    task_name = dtree.attrs.get("scan_name", "").strip()
    if not task_name:
        raise ValueError("Missing 'scan_name' in radar data attributes.")

    dtree = dtree.pipe(fix_angle).pipe(apply_georeferencing)
    dtree = handle_dynamic_vcp(dtree, engine=engine, append_dim=append_dim)

    new_dtree = DataTree.from_dict({task_name: dtree})
    new_dtree.encoding = dtree_encoding(new_dtree, append_dim=append_dim)
    return new_dtree


def handle_dynamic_vcp(dtree: DataTree, engine: str, append_dim: str) -> DataTree:
    if engine == "nexradlevel2" and check_dynamic_scan(dtree):
        return align_dynamic_scan(dtree, append_dim=append_dim)
    return ensure_dimension(dtree, append_dim=append_dim)
