from xarray import DataTree


def apply_georeferencing(dtree: DataTree) -> DataTree:
    """
    Apply georeferencing using xradar to each sweep in the DataTree.

    Parameters:
    -----------
    dtree : DataTree
        The radar data organized in a DataTree.

    Returns:
    --------
    DataTree
        The georeferenced DataTree.
    """
    return dtree.xradar.georeference()
