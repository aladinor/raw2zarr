import json
from pathlib import Path


def load_json_config(filename: str = "vcp.json") -> dict:
    """
    Load and parse a JSON configuration file from raw2zarr/config/.

    Parameters:
        filename (str): JSON file name (e.g. 'vcp.json')

    Returns:
        dict: Parsed contents of the JSON file.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be parsed.
    """
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config_path = config_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{filename}' not found at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {filename}: {e}")
