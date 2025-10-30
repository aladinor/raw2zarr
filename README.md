<img src="images/radar_datatree.png" alt="thumbnail" width="550"/>

# FAIR open radar data
[![DOI](https://zenodo.org/badge/658848435.svg)](https://zenodo.org/doi/10.5281/zenodo.10069535)
[![GitHub Release](https://img.shields.io/github/v/release/aladinor/raw2zarr?display_name=tag&sort=semver)](https://github.com/aladinor/raw2zarr/releases/latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aladinor/raw2zarr/main)

## 📄 Cite This Work

This repository implements the **Radar DataTree** framework described in:

**Ladino-Rincón & Nesbitt (2025)**
*Radar DataTree: A FAIR and Cloud-Native Framework for Scalable Weather Radar Archives*
arXiv:2510.24943 — https://arxiv.org/abs/2510.24943

Please cite this work if you use Raw2Zarr or the Radar DataTree model.

## Motivation

Weather radar data are among the most scientifically valuable yet structurally underutilized Earth observation datasets. Radars are vital in meteorology, detecting severe weather early and enabling timely warnings, saving lives, and reducing property damage. Beyond real-time forecasting, radar data supports critical applications including statistical analysis, climatology, and long-term atmospheric research.

Despite widespread public availability, operational radar archives remain **fragmented, vendor-specific, and poorly aligned with FAIR principles** (Findable, Accessible, Interoperable, Reusable). Traditional radar data storage involves millions of standalone binary files in proprietary or semi-standardized formats designed for real-time operations, not scientific reuse. Each radar volume scan, comprising data collected through multiple cone-like sweeps at various elevation angles, is stored as an individual file every 5-10 minutes. This file-centric model creates significant barriers: no temporal indexing, inconsistent metadata encoding, and extensive preprocessing required for analysis at scale.

**Radar DataTree** addresses these limitations by transforming operational radar archives into FAIR-compliant, cloud-optimized datasets. This framework extends the WMO [FM-301/CfRadial 2.1](https://community.wmo.int/en/activity-areas/wis/wmo-cf-extensions) standard from individual radar volume scans to **time-resolved, analysis-ready archives**. Built on **xarray.DataTree** for hierarchical data representation and **Icechunk** for ACID-compliant transactional storage, Radar DataTree enables:

- **Dataset-level organization**: Entire radar archives as structured, time-indexed collections
- **Cloud-native access**: Zarr serialization optimized for parallel I/O and lazy evaluation
- **Metadata preservation**: Full FM-301/CF compliance with sweep-level detail
- **Concurrent-safe writes**: Icechunk transactions enable real-time ingestion without data corruption
- **Scalable performance**: Demonstrated 100x+ speedups over traditional file-based workflows

This approach leverages a modern Python ecosystem including Xarray, Xradar, Wradlib, and Zarr to implement a hierarchical, tree-like data model aligned with Analysis-Ready Cloud-Optimized (ARCO) principles and FAIR data stewardship.


## Authors

[Alfonso Ladino-Rincon](https://github.com/aladinor),
[Max Grover](https://github.com/mgrover1)

### Collaborators

<a href="https://github.com/aladinor/raw2zarr/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aladinor/raw2zarr" />
</a>


```{warning}
This project is currently in high development mode.

Features may change frequently, and some parts of the library may be incomplete or subject to change.
Proceed with caution.
```

```{caution}
Critical storage requirements

Processing radar data can create massive storage requirements. Based on real-world experience:
- 1 month of data ≈ 800 GB of Zarr output
- 1 year of data ≈ 10+ TB of storage
- Processing for long periods can exhaust storage quickly

Recommendations:
- Start with hourly datasets to understand the storage footprint
- Plan storage capacity before production deployments
- Consider cloud storage costs for large-scale processing
- Monitor disk usage during runs
```

## Radar DataTree Framework

The Radar DataTree framework provides a **dataset-level abstraction** for weather radar collections, extending the WMO FM-301 standard from individual radar volume scans to time-resolved archives.

### Core Architecture

| Component | Role |
|-----------|------|
| **FM-301/CfRadial 2.1** | File-level standard for radar volumes and sweeps |
| **xarray.DataTree** | Hierarchical in-memory representation of scan collections |
| **Zarr** | Chunked, compressed, cloud-native storage format |
| **Icechunk** | ACID-compliant transactional engine for versioned datasets |

### Key Features

**Time-Indexed Collections**
Each radar archive is represented as a hierarchical tree of datasets aligned along a common time axis. Individual volume scans preserve their original FM-301 structure (sweep groups, coordinates, metadata) while being organized into a unified time-series dataset.

**Cloud-Native Storage**
Zarr serialization enables:
- Efficient partial reads and lazy evaluation
- Parallel I/O across distributed workers
- Compressed, chunked arrays optimized for cloud object storage

**ACID Transactions with Icechunk**
Icechunk provides:
- Safe concurrent writes from multiple workers
- Version-controlled datasets with atomic commits
- Real-time ingestion without data corruption
- Reproducible analysis with provenance tracking

**Demonstrated Performance**
Case studies on operational NEXRAD archives show:
- **100x+ speedup** for Quasi-Vertical Profile (QVP) generation
- **70-150x speedup** for Quantitative Precipitation Estimation (QPE)
- Sub-minute retrieval of multi-week time series from cloud storage

**Supported Formats**
- NEXRAD Level II (including dynamic scans: SAILS, MRLE, AVSET)
- SIGMET/IRIS
- ODIM_H5

### Demo Notebooks

Explore interactive examples at the [Radar DataTree Demo Repository](https://github.com/earth-mover/radar-data-demo):
- QVP computation from cloud-hosted archives
- QPE accumulation workflows
- Time-series extraction and analysis

## Getting Started

### Running on Your Own Machine
If you are interested in running this material locally on your computer, you will need to follow this workflow:

1. Clone the ["raw2zarr"](https://github.com/aladinor/raw2zarr) repository
    ```bash
    git clone https://github.com/aladinor/raw2zarr.git
    ```

2. Move into the `raw2zarr` directory
    ```bash
    cd raw2zarr
    ```

3. Create and activate your conda environment from the `environment.yml` file
    ```bash
    conda env create -f environment.yml
    conda activate raw2zarr
    ```

4.  Move into the `notebooks` directory and start up Jupyterlab
    ```bash
    cd notebooks/
    jupyter lab
    ```

## Processing Modes

The library supports two processing modes for converting radar data to Zarr format. Both modes use **Icechunk** for ACID-compliant transactional storage, ensuring data integrity during writes.

### Sequential Processing (No Cluster Required)

For small datasets, testing, and development:

```python
from raw2zarr.builder.convert import convert_files
from raw2zarr.builder.builder_utils import get_icechunk_repo

# Create repository
repo = get_icechunk_repo("output.zarr")

# Sequential processing
convert_files(
    radar_files=files,
    append_dim="vcp_time",
    repo=repo,
    process_mode="sequential",  # No cluster needed
    engine="nexradlevel2"
)
```

### Parallel Processing (Cluster Required)

For large datasets and production use. Uses **Icechunk's Session.fork()** API for concurrent-safe parallel writes:

```python
from dask.distributed import LocalCluster
from raw2zarr.builder.convert import convert_files
from raw2zarr.builder.builder_utils import get_icechunk_repo

# Create repository and cluster
repo = get_icechunk_repo("output.zarr")
cluster = LocalCluster(n_workers=4, memory_limit="10GB")

try:
    convert_files(
        radar_files=files,
        append_dim="vcp_time",
        repo=repo,
        process_mode="parallel",
        cluster=cluster,  # Required for parallel mode
        engine="nexradlevel2"
    )
finally:
    cluster.close()
```

**Performance**: Parallel processing with Icechunk enables:
- Concurrent writes from multiple workers without data corruption
- 100x+ speedups for large-scale radar analysis tasks
- Safe real-time ingestion alongside ongoing analysis
## References

* **Ladino-Rincón, A., & Nesbitt, S. W. (2025).** Radar DataTree: A FAIR and Cloud-Native Framework for Scalable Weather Radar Archives. *arXiv preprint arXiv:2510.24943*. https://arxiv.org/abs/2510.24943

* Abernathey, R. P., et al. (2021). Cloud-Native Repositories for Big Scientific Data. *Computing in Science & Engineering*, 23(2), 26-35. doi:10.1109/MCSE.2021.3059437
