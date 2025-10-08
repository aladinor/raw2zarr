import os
import warnings
from collections.abc import Iterable

from xarray import DataTree
from xarray.backends.common import _normalize_path

from ..io.load import load_radar_data
from ..templates.vcp_utils import map_sweeps_to_vcp_indices
from ..transform.alignment import fix_angle
from ..transform.dimension import ensure_dimension, slice_to_vcp_dimensions
from ..transform.encoding import dtree_encoding
from ..transform.georeferencing import apply_georeferencing
from .builder_utils import remove_dims


def radar_datatree(
    filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
    engine: str = "iris",
    append_dim: str = "vcp_time",
    is_dynamic: bool = False,
    sweep_indices: list[int] | None = None,
    elevation_angles: list[float] | None = None,
    vcp_config_file: str = "vcp_nexrad.json",
) -> DataTree:
    """
    Load and transform radar file(s) into a hierarchical xarray.DataTree.

    This function reads one or more radar files and constructs a DataTree,
    applying angle correction, georeferencing, and dimension alignment.
    Dynamic scanning modes (SAILS/MRLE/AVSET) use template-based processing
    to ensure consistent structure with NaN-filled missing sweeps.

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
        is_dynamic (bool, optional):
            Whether to use template-based processing for dynamic scans.
            Set True for AVSET, SAILS, MRLE, MESO-SAILS scans.
        sweep_indices (list[int] | None, optional):
            Indices of sweeps to include when processing temporal slices from
            dynamic scans (SAILS, MRLE, MESO-SAILS). If None, all sweeps are
            included (standard scan or AVSET behavior).
        scan_type (str | None, optional):
            Scan type classification for metadata and diagnostics.
            Examples: "STANDARD", "AVSET", "SAILS", "MRLE×3", "MESO-SAILS×2"
        elevation_angles (list[float] | None, optional):
            Elevation angles for this temporal slice (for VCP sweep index mapping).
            Used to map loaded sweeps to correct VCP sweep indices.
        vcp_config_file (str, optional):
            VCP configuration file name (default: "vcp_nexrad.json")

    Returns:
        DataTree:
            A hierarchical container of radar datasets, where each node represents
            a processed radar sweep or metadata component.

    Raises:
        ValueError: If the scan name is missing or file loading fails.

    Example:
        Standard scan:
        >>> tree = radar_datatree("file.RAW", engine="iris")
        >>> tree["sweep_0"].to_dataset()

        Dynamic scan temporal slice:
        >>> tree = radar_datatree(
        ...     "SAILS_file.RAW",
        ...     engine="nexradlevel2",
        ...     is_dynamic=True,
        ...     sweep_indices=[0,1,2,3,4,5],
        ...     scan_type="SAILS"
        ... )
    """

    filename_or_obj = _normalize_path(filename_or_obj)
    dtree = load_radar_data(filename_or_obj, engine=engine)
    if engine == "iris":
        dtree = remove_dims(dtree, "sweep")
    task_name = dtree.attrs.get("scan_name", "").strip()
    if not task_name:
        warnings.warn("Missing 'scan_name' in radar data attributes", UserWarning)

    if engine == "nexradlevel2":
        dtree = slice_to_vcp_dimensions(dtree, task_name, vcp_config_file)

    dtree = dtree.pipe(fix_angle).pipe(apply_georeferencing)

    if is_dynamic:
        dtree = map_sweeps_to_vcp_indices(
            data_tree=dtree,
            vcp=task_name,
            sweep_indices=sweep_indices,
            elevation_angles=elevation_angles,
            vcp_config_file=vcp_config_file,
        )

    dtree = ensure_dimension(dtree, append_dim=append_dim)
    if task_name:
        new_dtree = DataTree.from_dict({task_name: dtree})
    else:
        new_dtree = DataTree.from_dict({"DEFAULT": dtree})
    new_dtree.encoding = dtree_encoding(new_dtree, append_dim=append_dim)
    return new_dtree
