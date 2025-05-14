from ..config.utils import load_json_config


def get_vcp_values(
    vcp_name: str = "VCP-212", config_file: str = "vcp.json"
) -> list[float]:
    """
    Load elevation angles for a given Volume Coverage Pattern (VCP).

    Parameters:
        vcp_name (str): Name of the VCP (e.g., "VCP-212").
        config_file (str): Path to the JSON config file.

    Returns:
        list[float]: Elevation angles in degrees.

    Raises:
        KeyError: If the VCP name is not found.
        ValueError: If the structure is invalid.
    """
    config = load_json_config(config_file)

    try:
        elevations = config[vcp_name]["elevations"]
    except KeyError as e:
        raise KeyError(f"VCP '{vcp_name}' not found in {config_file}.") from e

    if not isinstance(elevations, list) or not all(
        isinstance(e, (int, float)) for e in elevations
    ):
        raise ValueError(f"Invalid 'elevations' list for {vcp_name}: {elevations}")

    return elevations


def _get_missing_elevations(
    default_list: list, second_list: list, tolerance: float = 0.05
) -> list[float]:
    i = 0
    j = 0
    while i < len(default_list) and j < len(second_list):
        if abs(default_list[i] - second_list[j]) <= tolerance:
            i += 1
            j += 1
        else:
            j += 1
    return [idx for idx in range(i, len(default_list))]
