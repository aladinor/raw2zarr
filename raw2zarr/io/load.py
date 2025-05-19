from raw2zarr.io.loaders.iris import iris_loader
from raw2zarr.io.loaders.nexrad import nexradlevel2_loader

ENGINE_REGISTRY = {
    "iris": iris_loader,
    "nexradlevel2": nexradlevel2_loader,
}


def load_radar_data(file, engine="iris"):
    """
    Unified loader interface for radar formats.

    Parameters:
        file (str or file-like or S3File): Input file path or object
        engine (str): One of "iris", "nexradlevel2"

    Returns:
        xarray.DataTree: Parsed radar data
    """
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unsupported engine '{engine}'. Supported: {list(ENGINE_REGISTRY.keys())}"
        )

    return ENGINE_REGISTRY[engine](file)
