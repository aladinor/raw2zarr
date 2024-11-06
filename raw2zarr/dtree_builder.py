from typing import List, Iterable, Union
import os
import xarray as xr
from xarray import DataTree, Dataset
from xarray.backends.common  import _normalize_path
from data_reader import load_radar_data  # Import the data loading function
from raw2zarr.utils import ensure_dimension

def datatree_builder(
        filename_or_obj: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
        backend: str = "iris",
        dim: str = "vcp_time",
        parallel: bool = False,
        batch_size: int = 12
) -> DataTree:
    """
    Load radar data from files in batches and build a nested DataTree from it.

    Parameters:
        filename_or_obj (str | os.PathLike | Iterable[str | os.PathLike]): Path(s) to radar data files.
        backend (str): Backend type to use. Options include 'iris', 'odim', etc. Default is 'iris'.
        parallel (bool): If True, enables parallel processing with Dask. Default is False.
        batch_size (int): Number of files to process in each batch.

    Returns:
        DataTree: A nested DataTree combining all input DataTree objects.

    Raises:
        ValueError: If no files are loaded or all batches are empty.
    """
    # Initialize an empty dictionary to hold the nested structure
    nested_dict = {}

    # Load radar data in batches
    filename_or_obj = _normalize_path(filename_or_obj)

    for dtree_batch in load_radar_data(filename_or_obj, backend=backend, parallel=parallel, batch_size=batch_size):
        if not dtree_batch:
            raise ValueError("A batch of DataTrees is empty. Ensure data is loaded correctly.")

        # Process each DataTree in the current batch
        for dtree in dtree_batch:
            task_name = dtree.attrs.get("scan_name", "default_task").strip()

            if task_name in nested_dict:
               nested_dict[task_name] = append_dataset_to_node(nested_dict[task_name], dtree, dim=dim)
            else:
                nested_dict[task_name] = dtree

    # Final DataTree assembly
    return DataTree.from_dict(nested_dict)


def append_dataset_to_node(existing_node: DataTree, new_node: DataTree, dim: str):
    """
    Append datasets from new_node to the existing_node's structure.

    Parameters:
        existing_node (DataTree): The existing node in the nested DataTree to which data will be appended.
        new_node (DataTree): The new DataTree node containing datasets to be appended.
    """

    existing_node = ensure_dimension(existing_node, dim)
    new_node = ensure_dimension(new_node, dim)
    new_dtree = {}
    for child in new_node.subtree:
        node_name = child.path
        if node_name in [node.path for node in existing_node.subtree]:
            # Append the datasets if the node already exists
            existing_dataset = existing_node[node_name].to_dataset()
            new_dataset = child.to_dataset()
            # Append data along a new dimension (e.g., time) or merge variables as needed
            new_dtree[node_name] = xr.concat((existing_dataset, new_dataset), dim=dim)
        else:
            new_dtree = new_node[node_name].to_dataset()

    return DataTree.from_dict(new_dtree)

