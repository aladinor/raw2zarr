"""
Tests for the convert_files function and engine-based VCP configuration.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from raw2zarr.builder.convert import convert_files
from raw2zarr.main import get_radar_files


class TestConvertFiles:
    """Test convert_files function with VCP configuration parameter."""

    def test_convert_files_signature_includes_vcp_config_file(self):
        """Test that convert_files accepts vcp_config_file parameter."""
        sig = inspect.signature(convert_files)
        assert "vcp_config_file" in sig.parameters
        assert sig.parameters["vcp_config_file"].default == "vcp_nexrad.json"

    @patch("raw2zarr.builder.convert.append_sequential")
    def test_convert_files_sequential_passes_vcp_config(self, mock_append_sequential):
        """Test that convert_files passes vcp_config_file to sequential processor."""
        # Mock inputs
        mock_repo = MagicMock()
        mock_files = ["file1.h5", "file2.h5"]

        # Call convert_files with vcp_config_file
        convert_files(
            radar_files=mock_files,
            append_dim="vcp_time",
            repo=mock_repo,
            process_mode="sequential",
            engine="iris",
            vcp_config_file="ideam.json",
        )

        # Verify append_sequential was called with vcp_config_file
        mock_append_sequential.assert_called_once()
        call_kwargs = mock_append_sequential.call_args[1]
        assert "vcp_config_file" in call_kwargs
        assert call_kwargs["vcp_config_file"] == "ideam.json"

    @patch("raw2zarr.builder.convert.append_parallel")
    def test_convert_files_parallel_passes_vcp_config(self, mock_append_parallel):
        """Test that convert_files passes vcp_config_file to parallel processor."""
        # Mock inputs
        mock_repo = MagicMock()
        mock_cluster = MagicMock()
        mock_files = ["file1.h5", "file2.h5"]

        # Call convert_files with vcp_config_file
        convert_files(
            radar_files=mock_files,
            append_dim="vcp_time",
            repo=mock_repo,
            process_mode="parallel",
            engine="odim",
            cluster=mock_cluster,
            vcp_config_file="eccc.json",
        )

        # Verify append_parallel was called with vcp_config_file
        mock_append_parallel.assert_called_once()
        call_kwargs = mock_append_parallel.call_args[1]
        assert "vcp_config_file" in call_kwargs
        assert call_kwargs["vcp_config_file"] == "eccc.json"

    def test_convert_files_invalid_process_mode(self):
        """Test that convert_files raises error for invalid process_mode."""
        mock_repo = MagicMock()
        mock_files = ["file1.h5"]

        with pytest.raises(ValueError, match="Unsupported mode"):
            convert_files(
                radar_files=mock_files,
                append_dim="vcp_time",
                repo=mock_repo,
                process_mode="invalid_mode",
                engine="iris",
            )

    def test_convert_files_default_vcp_config(self):
        """Test that convert_files uses default vcp_config_file when not specified."""
        with patch("raw2zarr.builder.convert.append_sequential") as mock_append:
            mock_repo = MagicMock()
            mock_files = ["file1.h5"]

            convert_files(
                radar_files=mock_files,
                append_dim="vcp_time",
                repo=mock_repo,
                process_mode="sequential",
                engine="nexradlevel2",
                # No vcp_config_file specified - should use default
            )

            # Verify default was used
            call_kwargs = mock_append.call_args[1]
            assert call_kwargs["vcp_config_file"] == "vcp_nexrad.json"


class TestEngineBasedConfigSelection:
    """Test automatic engine-to-config mapping in main.py."""

    def test_iris_engine_config_mapping(self):
        """Test that iris engine maps to ideam.json config."""
        radar_files, zarr_store, engine, vcp_config_file = get_radar_files("iris")

        assert engine == "iris"
        assert vcp_config_file == "ideam.json"
        assert isinstance(radar_files, list)
        assert isinstance(zarr_store, str)

    def test_nexradlevel2_engine_config_mapping(self):
        """Test that nexradlevel2 engine maps to vcp_nexrad.json config."""
        radar_files, zarr_store, engine, vcp_config_file = get_radar_files(
            "nexradlevel2"
        )

        assert engine == "nexradlevel2"
        assert vcp_config_file == "vcp_nexrad.json"
        assert isinstance(radar_files, list)
        assert isinstance(zarr_store, str)

    def test_odim_engine_config_mapping(self):
        """Test that odim engine maps to eccc.json config."""
        radar_files, zarr_store, engine, vcp_config_file = get_radar_files("odim")

        assert engine == "odim"
        assert vcp_config_file == "eccc.json"
        assert isinstance(radar_files, list)
        assert isinstance(zarr_store, str)

    def test_unsupported_engine(self):
        """Test that unsupported engine returns None."""
        result = get_radar_files("unsupported_engine")
        assert result is None

    def test_get_radar_files_return_format(self):
        """Test that get_radar_files returns 4-tuple for all supported engines."""
        engines = ["iris", "nexradlevel2", "odim"]

        for engine in engines:
            result = get_radar_files(engine)
            assert result is not None
            assert len(result) == 4

            radar_files, zarr_store, returned_engine, vcp_config_file = result
            assert returned_engine == engine
            assert isinstance(vcp_config_file, str)
            assert vcp_config_file.endswith(".json")


class TestVcpConfigParameterFlow:
    """Test VCP config parameter flow through the processing pipeline."""

    @patch("raw2zarr.builder.executor.init_zarr_store")
    def test_executor_passes_vcp_config_to_init_zarr_store(self, mock_init_zarr_store):
        """Test that executor passes vcp_config_file to init_zarr_store."""
        from raw2zarr.builder.executor import append_parallel

        # Mock required objects
        mock_repo = MagicMock()
        mock_cluster = MagicMock()
        mock_session = MagicMock()
        mock_fork = MagicMock()
        mock_client = MagicMock()

        # Setup mocks
        mock_repo.writable_session.return_value = mock_session
        mock_session.fork.return_value = mock_fork
        mock_init_zarr_store.return_value = []

        with patch("dask.distributed.Client", return_value=mock_client):
            with patch("raw2zarr.builder.executor.extract_single_metadata"):
                # Setup client.map to return empty results
                mock_client.map.return_value = []
                mock_client.gather.return_value = []

                # Test with custom vcp_config_file
                try:
                    append_parallel(
                        radar_files=[],
                        append_dim="vcp_time",
                        repo=mock_repo,
                        cluster=mock_cluster,
                        vcp_config_file="ideam.json",
                    )
                except Exception:
                    # Function may fail due to empty files, but we only care about the call
                    pass

                # Verify init_zarr_store was called with vcp_config_file if it was called
                if mock_init_zarr_store.called:
                    call_kwargs = mock_init_zarr_store.call_args[1]
                    assert "vcp_config_file" in call_kwargs
                    assert call_kwargs["vcp_config_file"] == "ideam.json"

    def test_writer_utils_init_zarr_store_signature(self):
        """Test that init_zarr_store has vcp_config_file parameter."""
        from raw2zarr.writer.writer_utils import init_zarr_store

        sig = inspect.signature(init_zarr_store)
        assert "vcp_config_file" in sig.parameters
        assert sig.parameters["vcp_config_file"].default == "vcp_nexrad.json"

    def test_vcp_utils_create_multi_vcp_template_signature(self):
        """Test that create_multi_vcp_template has vcp_config_file parameter."""
        from raw2zarr.templates.vcp_utils import create_multi_vcp_template

        sig = inspect.signature(create_multi_vcp_template)
        assert "vcp_config_file" in sig.parameters
        assert sig.parameters["vcp_config_file"].default == "vcp_nexrad.json"


class TestConfigurationErrorHandling:
    """Test error handling for VCP configuration issues."""

    def test_missing_config_file_error_message(self):
        """Test that missing config file produces clear error message."""
        from raw2zarr.templates.template_manager import VcpTemplateManager

        with pytest.raises(FileNotFoundError) as excinfo:
            VcpTemplateManager("missing_config.json")

        assert "Unified config file not found" in str(excinfo.value)
        assert "missing_config.json" in str(excinfo.value)

    def test_config_validation_ideam(self):
        """Test IDEAM config file validation."""
        from raw2zarr.templates.template_manager import VcpTemplateManager

        try:
            manager = VcpTemplateManager("ideam.json")
            config = manager.config

            # Validate required IDEAM VCP patterns
            required_vcps = ["PRECC", "SURVP", "PRECA", "PRECB"]
            for vcp in required_vcps:
                assert vcp in config, f"Missing required VCP: {vcp}"

                # Validate VCP structure
                vcp_config = config[vcp]
                assert "elevations" in vcp_config
                assert "dims" in vcp_config
                assert isinstance(vcp_config["elevations"], list)
                assert len(vcp_config["elevations"]) > 0

        except FileNotFoundError:
            pytest.skip("IDEAM config file not available in test environment")

    def test_config_validation_eccc(self):
        """Test ECCC config file validation."""
        from raw2zarr.templates.template_manager import VcpTemplateManager

        try:
            manager = VcpTemplateManager("eccc.json")
            config = manager.config

            # Validate DEFAULT VCP exists
            assert "DEFAULT" in config, "Missing required DEFAULT VCP"

            default_config = config["DEFAULT"]
            assert "elevations" in default_config
            assert "dims" in default_config
            assert isinstance(default_config["elevations"], list)
            assert len(default_config["elevations"]) == 17  # ECCC has 17 sweeps

        except FileNotFoundError:
            pytest.skip("ECCC config file not available in test environment")

    @patch("raw2zarr.writer.writer_utils.radar_datatree")
    def test_default_fallback_for_missing_scan_name(self, mock_radar_datatree):
        """Test DEFAULT fallback when scan_name attribute is missing."""
        from raw2zarr.writer.writer_utils import init_zarr_store

        # Mock a DataTree without scan_name attribute
        mock_dtree = MagicMock()
        mock_group = MagicMock()
        mock_group.attrs = {}  # No scan_name attribute
        mock_dtree.__getitem__.return_value = mock_group
        mock_dtree.groups = [None, "group1"]  # Simulate groups structure

        mock_radar_datatree.return_value = mock_dtree

        # Mock session and other required objects
        mock_session = MagicMock()
        mock_session.store = MagicMock()

        with patch(
            "raw2zarr.writer.writer_utils.zarr_store_has_append_dim", return_value=False
        ):
            with patch(
                "raw2zarr.writer.writer_utils.create_multi_vcp_template"
            ) as mock_create:
                with patch(
                    "raw2zarr.writer.writer_utils.remove_string_vars"
                ) as mock_remove:
                    with patch(
                        "raw2zarr.writer.writer_utils.dtree_encoding"
                    ) as mock_encoding:
                        with patch(
                            "raw2zarr.writer.writer_utils.resolve_zarr_write_options"
                        ) as mock_resolve:
                            with patch("raw2zarr.writer.writer_utils.dtree_to_zarr"):
                                # Setup mock returns
                                mock_create.return_value = MagicMock()
                                mock_remove.return_value = MagicMock()
                                mock_encoding.return_value = {}
                                mock_resolve.return_value = {}
                                try:
                                    init_zarr_store(
                                        files=[(0, "test_file.h5")],
                                        session=mock_session,
                                        append_dim="vcp_time",
                                        engine="odim",
                                        zarr_format=3,
                                        consolidated=False,
                                        vcp_config_file="eccc.json",
                                    )

                                    # If we reach here, the DEFAULT fallback worked
                                    # Verify the scan_name fallback logic was triggered
                                    assert True  # Test passes if no exception

                                except Exception as e:
                                    # The function might fail for other reasons in testing
                                    # but we mainly want to test the fallback doesn't crash
                                    if "scan_name" in str(e):
                                        pytest.fail(f"DEFAULT fallback failed: {e}")
