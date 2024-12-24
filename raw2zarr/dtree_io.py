from __future__ import annotations
import os
import gzip
import tempfile

import xradar
import fsspec
from s3fs import S3File

from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal

from xarray import DataTree
from xarray.core.types import ZarrWriteModes

# Relative imports
from .utils import prepare_for_read


def accessor_wrapper(
    filename_or_obj: str | os.PathLike,
    engine: str = "iris",
) -> DataTree:
    """Wrapper function to load radar data for a single file or iterable of files with fsspec and compression check."""
    try:
        file = prepare_for_read(filename_or_obj)
        return _load_file(file, engine)
    except Exception as e:
        print(f"Error loading {filename_or_obj}: {e}")
        return None


def _load_file(file, engine) -> DataTree:
    """Helper function to load a single file with the specified backend."""
    if engine == "iris":
        if isinstance(file, S3File):
            return xradar.io.open_iris_datatree(file.read())
        elif isinstance(file, bytes):
            return xradar.io.open_iris_datatree(file)
        else:
            return xradar.io.open_iris_datatree(file)
    elif engine == "odim":
        return xradar.io.open_odim_datatree(file)
    elif engine == "nexradlevel2":
        if isinstance(file, S3File):
            local_file = fsspec.open_local(
                f"simplecache::s3://{file.path}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            data_tree = xradar.io.open_nexradlevel2_datatree(local_file)

            # Remove the local file after loading the data
            os.remove(local_file)
            return data_tree
        elif isinstance(file, gzip.GzipFile):
            local_file = fsspec.open_local(
                f"simplecache::s3://{file.fileobj.full_name}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            with gzip.open(local_file, "rb") as gz:
                # Step 2: Write the decompressed content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(gz.read())
                    temp_file_path = temp_file.name

                # Step 3: Use xradar to open the temporary file
            try:
                data_tree = xradar.io.open_nexradlevel2_datatree(temp_file_path)
            finally:
                # Step 4: Clean up the temporary files
                os.remove(local_file)
                os.remove(temp_file_path)
            return data_tree
        else:
            return xradar.io.open_nexradlevel2_datatree(file)
    else:
        raise ValueError(f"Unsupported backend: {engine}")


def _datatree_to_zarr(
    dt: DataTree,
    store: MutableMapping | str | PathLike[str],
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    consolidated: bool = True,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: Literal[True] = True,
    **kwargs,
):
    """This function creates an appropriate datastore for writing a datatree
    to a zarr store.

    See `DataTree.to_zarr` for full API docs.
    """

    from zarr import consolidate_metadata

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not compute:
        raise NotImplementedError("compute=False has not been implemented yet")

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )
    append_dim = kwargs.pop("append_dim", None)
    for node in dt.subtree:
        at_root = node is dt
        if node.is_empty & node.is_root:
            continue
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)
        group_path = None if at_root else "/" + node.relative_to(dt)
        try:
            ds.to_zarr(
                store,
                group=group_path,
                mode=mode,
                consolidated=False,
                append_dim=append_dim,
                **kwargs,
            )
        except ValueError as e:
            ds.to_zarr(
                store,
                group=group_path,
                mode="a-",
                encoding=encoding.get(node.path),
                consolidated=False,
                **kwargs,
            )

        if "w" in mode:
            mode = "a"

    if consolidated:
        consolidate_metadata(store)
