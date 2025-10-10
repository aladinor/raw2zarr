# Changelog

All notable changes to this project are documented here.

## [Unreleased]

Highlights
- Dynamic scan support with temporal slicing for NEXRAD (SAILS, MRLE, AVSET patterns)
- High-performance async S3 file listing utilities (10-100x speedup)
- Metadata processor with corruption detection and filtering
- VCP backward compatibility system for range dimension slicing
- Python 3.11 support for broader deployment compatibility

Breaking Changes
- None

Features
- Add async S3 file listing utilities with parallel day-level queries (6577d07, 809b7e2)
- Add metadata processor for parallel file processing with corruption filtering (c81584b)
- Add VCP sweep mapping (`map_sweeps_to_vcp_indices`) for dynamic scan handling (c81584b)
- Add VCP time mapping with temporal slice support for SAILS/MRLE/AVSET (c81584b)
- Add `slice_to_vcp_dimensions()` for VCP backward compatibility (b791af7)
- Add `create_sweep_to_vcp_mapping()` shared helper for elevation-based sweep mapping (e62a803)
- Add corruption detection for missing/misaligned sweep indices (c81584b)
- Add timezone-aware timestamp conversion utility (c81584b)
- Add Python 3.11 support across setup.py, pyproject.toml, and CI matrix (d6c538a)
- Add NCSA cluster deployment environment with HTCondor support (554aa2f)
- Implement dynamic scan writing with temporal slicing (c751f54, 90962a7)
- Add dynamic scan detection to `radar_datatree()` (299a250)

Improvements
- Refactor sweep-to-VCP mapping logic into reusable helper function (e62a803)
- Update VCP configurations using KLOT 2025 files (bac2bbc)
- Reorganize metadata fields: rename `dynamic_type` → `scan_type`, `sails_inserts` → `additional_sweeps` (eef1286)
- Add `missing_sweeps` field to metadata (eef1286)
- Improve metadata extraction placement for simplification (263877e)
- Filter MDM files from file queries (289746a, b50a3fd)
- Remove unused parameters (7ce7a80)

Fixes
- Fix filepath index mismatch when corrupted files are skipped in metadata processing (e62a803)
- Fix VCP-212 range dimensions for sweeps 10-13 to match 2025 data (e62a803)
- Fix dimension mismatch errors in parallel region writes for temporal slices (e62a803)
- Fix VCP range slicing and template creation TypeError in `create_scan_dataset()` (b791af7)
- Restrict VCP slicing to NEXRAD data only (b791af7)
- Fix file listing bug (f88af3e)
- Update test patches for utils package structure (b1fb2d2)
- Export private test helper functions from utils (fb8e0eb)
- Fix environment.yml CI conflicts by removing GitHub install (1f35f16)
- Use `python>=3.11` in environment.yml for CI matrix flexibility (100d94e)
- Update notebook-check CI job to use Python 3.11 (5e1201c)

Tests
- Add test for filepath mapping with corrupted files (e62a803)
- Add 30 comprehensive tests for metadata processing and corruption detection (c81584b)
- Add tests for VCP utilities (sweep mapping, time mapping with slices) (c81584b)
- Add tests for timezone conversion (UTC, naive, timezone-aware handling) (c81584b)
- Add 4 integration tests using real S3 NEXRAD files (c81584b)

Documentation
- Convert README admonitions to MyST format (6e00f1d)
- Add NEXRAD dynamic scans warning to documentation (6e00f1d)
- Add GitHub Release badge marking v0.4.0 as latest (5ec1dbb)

CI / Environment
- Add Python 3.11 to CI test matrix alongside Python 3.12 (d6c538a)
- Add NCSA-specific environment with HTCondor integration (554aa2f)
- Fix CI environment conflicts (1f35f16, 100d94e, 5e1201c)

Upgrade Notes
- Python 3.11+ now supported (previously required 3.12+)
- New async utilities available: use `get_radar_files_async()` for faster multi-day S3 queries
- VCP range slicing now automatic for NEXRAD data (backward compatibility with older VCP configs)
- Metadata field names changed: update code using `dynamic_type` or `sails_inserts`

Full diff: compare `v0.4.0...HEAD`.

## [0.4.0] - 2025-09-29

Highlights
- Multi-engine VCP configuration system across backends.
- Unified VCP config system with improved template handling.
- New ODIM backend support.
- New writer utility `check_cords` to ensure x, y, z, and time are coordinates at the sweep level.
- CI, notebooks, and environment stability improvements.

Breaking Changes
- Simplify config naming and remove backward compatibility in parts of the config system (refactor; d3cb5e1).
- Unified VCP config and template adjustments may require updating existing configs/templates (54f1014).

Features
- Implement multi-engine VCP configuration system (c0656e7).
- Add ODIM backend (39a2ac6).
- Add `check_cords` utility to enforce coordinate placement (97faf79, 574917c).
- API enhancements (6da1f40).

Improvements
- Properly encode `append_dim` in dtree encoding (556314f).
- Improve dtree root naming when task/scan name is missing (943c965).
- Optimize builder module; refactor utils with cleanup and new NEXRAD utility (ad31398, ee19405).
- Remove deprecated `radar_wms_original.py` module (453e98e).

Fixes
- Resolve template dimension issues (2e3ace4).
- Fix TypeError (7809dbc).
- Fix default dict bug (84c22e2).
- Updates to sampler verification and paths (a0a4046, 79c296d, 8e9eadd).
- Lint and small cleanups (46d0dca, 96316e1, b3d383a).

CI / Notebooks / Docs
- Notebook CI improvements, nbqa config, and error handling (1de664b, 08df814, 6c8d8c1, 081f280, 4a3bb32).
- Add QVP notebook to tests and improve notebooks (38c80b5, 1775b21, f9be326).
- Numpy version compatibility and pinning for CI stability (6e04a54, 90c5996, ada4401).

Dependencies / Environment
- Add `cmweather` dependency (38fa0a4, 565e5dd, 0ff9234, 525e5d7, cba9810).
- Update environment and dev dependencies; add lint/test dependencies (bb741f7, aed9bca, ed0dd1b).

Tests
- Add coordinate sanity check test (3f2de2d).
- Add tests for IDEAM and ECCC JSON files (41dd116, c1951f3).
- Various test updates and fixes (77b9448, 39f9204, 1125e91).

Upgrade Notes
- Review config changes under `raw2zarr/config/` and template usage; update VCP configuration references due to the unified system.
- If using older config names, adjust to the simplified scheme introduced in 0.4.0.

Full diff: compare `v0.3.0...v0.4.0`.
