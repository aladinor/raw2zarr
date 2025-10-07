"""
Metadata processing and VCP mapping creation for parallel file conversion.

This module handles the workflow of extracting, filtering, and organizing
metadata from radar files for parallel processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.distributed import Client

from .builder_utils import _log_problematic_file, generate_vcp_samples


class MetadataProcessingResult:
    """Result of metadata processing workflow."""

    def __init__(
        self,
        vcp_time_mapping: dict,
        valid_files: list[tuple[int, str]],
        problematic_count: int,
        skipped_vcp_count: int,
    ):
        self.vcp_time_mapping = vcp_time_mapping
        self.valid_files = valid_files
        self.problematic_count = problematic_count
        self.skipped_vcp_count = skipped_vcp_count

    @property
    def vcp_names(self) -> list[str]:
        return list(self.vcp_time_mapping.keys())

    @property
    def total_valid_files(self) -> int:
        return sum(info["file_count"] for info in self.vcp_time_mapping.values())


def flatten_metadata_results(metadata_results_nested):
    """
    Flatten nested metadata results from extract_single_metadata().

    Dynamic scans return lists of tuples, standard scans return single tuple.
    This flattens everything into a single list.

    Parameters
    ----------
    metadata_results_nested : list
        List of results from extract_single_metadata(), where each result can be:
        - A list of tuples (for dynamic scans with multiple temporal slices)
        - A single tuple (legacy behavior for standard scans)

    Returns
    -------
    list of tuple
        Flattened list where each entry is:
        (original_index, file, timestamp, vcp, slice_id, sweep_indices, scan_type)
    """
    flattened = []
    for result in metadata_results_nested:
        if isinstance(result, list):
            flattened.extend(result)  # Dynamic scan with multiple slices
        else:
            flattened.append(result)  # Legacy single tuple
    return flattened


def process_metadata_and_create_vcp_mapping(
    client: Client,
    radar_files: list,
    engine: str,
    skip_vcps: list[str] | None = None,
    log_file: str | None = None,
    generate_samples: bool = False,
    sample_percentage: float = 15.0,
    samples_output_path: str | None = None,
) -> MetadataProcessingResult | None:
    """
    Extract metadata, filter files, and create VCP time mapping.

    This function orchestrates the complete metadata processing workflow:
    1. Parallel metadata extraction from radar files
    2. Flattening nested results (for dynamic scans)
    3. Filtering problematic files and unwanted VCPs
    4. Re-indexing temporal slices
    5. Creating VCP time mapping
    6. Logging and reporting
    7. Optional validation sample generation

    Parameters
    ----------
    client : Client
        Dask distributed client for parallel processing
    radar_files : list
        List of radar file paths
    engine : str
        Radar file engine (e.g., "nexradlevel2", "iris")
    skip_vcps : list[str] | None
        VCP patterns to skip (e.g., ["VCP-31", "VCP-32"])
    log_file : str | None
        Path to log file for problematic files
    generate_samples : bool
        Whether to generate VCP validation samples
    sample_percentage : float
        Percentage of files to sample for validation
    samples_output_path : str | None
        Path to save VCP samples JSON

    Returns
    -------
    MetadataProcessingResult | None
        Processing result with VCP mapping and file lists, or None if no valid files
    """
    from ..templates.vcp_utils import create_vcp_time_mapping_with_slices
    from .builder_utils import extract_single_metadata

    # 1. Parallel metadata extraction
    radar_files_with_indices = list(enumerate(radar_files))
    futures = client.map(
        extract_single_metadata, radar_files_with_indices, engine=engine
    )
    metadata_results_nested = client.gather(futures)

    # 2. Flatten results (dynamic scans return lists of tuples)
    metadata_results = flatten_metadata_results(metadata_results_nested)

    # 3. Filter and re-index
    valid_results = []
    valid_files = []
    problematic_files = []
    skipped_vcps = []

    for slice_index, (
        original_index,
        file,
        timestamp,
        vcp,
        slice_id,
        sweep_indices,
        scan_type,
        elevation_angles,
    ) in enumerate(metadata_results):
        if timestamp != "ERROR":
            # Check if VCP should be skipped
            if skip_vcps and vcp in skip_vcps:
                skipped_vcps.append((file, vcp))
                continue

            valid_results.append(
                (timestamp, vcp, slice_id, sweep_indices, scan_type, elevation_angles)
            )
            valid_files.append((slice_index, file))
        else:
            # Problematic file (vcp contains error message)
            problematic_files.append((file, vcp))

    # 4. Log problematic files
    for file, error_msg in problematic_files:
        _log_problematic_file(file, error_msg, log_file)

    for file, vcp_name in skipped_vcps:
        _log_problematic_file(
            file, f"Skipped {vcp_name} (configured to skip)", log_file
        )

    # 5. Early return if no valid files
    if not valid_results:
        print("❌ No valid files found after filtering problematic files.")
        return None

    # 6. Report statistics
    _report_filtering_statistics(
        total_files=len(radar_files),
        valid_count=len(valid_results),
        problematic_count=len(problematic_files),
        skipped_vcp_count=len(skipped_vcps),
        skip_vcps=skip_vcps,
    )

    # 7. Create VCP time mapping
    vcp_time_mapping = create_vcp_time_mapping_with_slices(valid_results, valid_files)

    # 8. Report VCP discovery
    _report_vcp_discovery(vcp_time_mapping)

    # 9. Generate validation samples (optional)
    if generate_samples:
        generate_vcp_samples(
            vcp_time_mapping=vcp_time_mapping,
            sample_percentage=sample_percentage,
            output_path=samples_output_path,
        )

    return MetadataProcessingResult(
        vcp_time_mapping=vcp_time_mapping,
        valid_files=valid_files,
        problematic_count=len(problematic_files),
        skipped_vcp_count=len(skipped_vcps),
    )


def _report_filtering_statistics(
    total_files: int,
    valid_count: int,
    problematic_count: int,
    skipped_vcp_count: int,
    skip_vcps: list[str] | None,
) -> None:
    """Report file filtering statistics to console."""
    total_skipped = total_files - valid_count

    if total_skipped > 0:
        if skipped_vcp_count > 0:
            print(f"Skipped {skipped_vcp_count} files from filtered VCPs: {skip_vcps}")

        if problematic_count > 0:
            print(
                f"Filtered out {problematic_count} problematic files (see output.txt for details)"
            )

        print(f"Processing {valid_count} valid files")


def _report_vcp_discovery(vcp_time_mapping: dict) -> None:
    """Report discovered VCP patterns to console."""
    vcp_names = list(vcp_time_mapping.keys())
    total_files = sum(info["file_count"] for info in vcp_time_mapping.values())

    print(f"Discovered {len(vcp_names)} VCP patterns in {total_files} files:")
    print(f"VCPs found: {', '.join(vcp_names)}")
