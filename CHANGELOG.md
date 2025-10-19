# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- **VCP Config**: Changed `follow_mode`, `prt_mode`, and `sweep_mode` dtypes from `int32` to `U50` to match actual xradar string data types ([#ccda1b9](https://github.com/aladinor/raw2zarr/commit/ccda1b9))
- **Templates**: Excluded scalar variables from 3D array creation - these variables now correctly have only `(vcp_time,)` dimension instead of `(vcp_time, azimuth, range)` ([#6ed437a](https://github.com/aladinor/raw2zarr/commit/6ed437a))
- **Templates**: Added `follow_mode` and `prt_mode` scalar variables to sweep templates to ensure parallel mode matches sequential mode structure ([#5be63a7](https://github.com/aladinor/raw2zarr/commit/5be63a7))
- **Templates**: Materialize VCP root variables before template write using `.compute()` to persist scalar metadata values (longitude, latitude, altitude) ([#f232ab0](https://github.com/aladinor/raw2zarr/commit/f232ab0))
- **Writer**: Cast string variables to U50 for template compatibility during parallel region writes ([#b53be83](https://github.com/aladinor/raw2zarr/commit/b53be83))
- **CI**: Added explicit conda-forge channel configuration to micromamba setup to fix package resolution failures ([#a7d2ff3](https://github.com/aladinor/raw2zarr/commit/a7d2ff3))
- **Environment**: Removed duplicate `pandas` entry and added version pins to pip-installed packages (`icechunk>=1.1.9`, `arraylake>=0.25`) ([#15805a6](https://github.com/aladinor/raw2zarr/commit/15805a6))

### Changed
- **Architecture**: Moved template operations (`create_vcp_template_in_memory`, `merge_data_into_template`) from `builder/executor.py` to new `templates/template_ops.py` module for better separation of concerns ([#a7b1a99](https://github.com/aladinor/raw2zarr/commit/a7b1a99), [#667c064](https://github.com/aladinor/raw2zarr/commit/667c064))

### Added
- **Tests**: Comprehensive unit tests for `template_ops.py` module (14 tests covering template creation and data merging) ([#1b4a9e0](https://github.com/aladinor/raw2zarr/commit/1b4a9e0))
- **Tests**: Enhanced parallel vs sequential equivalence test with parametrization for `remove_strings` and template placeholder handling ([#4c1d1da](https://github.com/aladinor/raw2zarr/commit/4c1d1da))
- **Tests**: Added U50 string dtype to config validation test's valid dtypes list ([#e64b0dd](https://github.com/aladinor/raw2zarr/commit/e64b0dd))

### Performance
- **Zarr**: Added async concurrency configuration (`async.concurrency: 24`) for improved parallel write performance ([#0d0005c](https://github.com/aladinor/raw2zarr/commit/0d0005c))

## [0.4.1] - 2025-10-15

**License Change**
- Changed license from BSD-3-Clause to CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)
- Added NOTICE file explaining license terms and commercial licensing options
- Updated pyproject.toml and setup.py with correct license metadata

**Documentation**
- Reorganized documentation files: moved SCIPY_POSTER.pdf to /docs folder
- Moved radar_FAIR.png to /images folder

**Important Notice**
This version restricts usage to non-commercial purposes only. For commercial licensing inquiries, contact alfonso8@illinois.edu.

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
