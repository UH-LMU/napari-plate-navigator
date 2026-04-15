"""
Integration tests for napari-plate-navigator.

These tests require real microscopy data. They are automatically skipped
if no data directory is provided. See conftest.py for how to supply data.

Run with:
    pytest --data-dir /path/to/testdata
    TEST_DATA_DIR=/path/to/testdata pytest
    TEST_DATA_URL=https://a3s.fi/bucket/testdata pytest
"""

import dask.array as da
import pytest

from napari_plate_navigator._reader import load_plate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_plate(result):
    """Common assertions for a load_plate result."""
    assert result["df"] is not None and not result["df"].empty
    plate = result["plate"]
    assert plate.nwells() > 0
    # Check at least one site has loadable images
    for well_name, well in plate.wells.items():
        for site_name, site in well.sites.items():
            images = site.get_images()
            assert len(images) > 0
            for name, arr in images.items():
                assert isinstance(arr, da.Array), (
                    f"Expected dask array for {well_name}/{site_name}/{name}"
                )
            return  # One site is enough for a smoke test


# ---------------------------------------------------------------------------
# ImageXpress
# ---------------------------------------------------------------------------

class TestImageXpressIntegration:
    def test_load_plate(self, imagexpress_dir, requires_data):
        if imagexpress_dir is None:
            pytest.skip("No imagexpress/ subdir in test data")
        result = load_plate(imagexpress_dir)
        _assert_valid_plate(result)

    def test_multiple_wells(self, imagexpress_dir, requires_data):
        if imagexpress_dir is None:
            pytest.skip("No imagexpress/ subdir in test data")
        result = load_plate(imagexpress_dir)
        assert result["plate"].nwells() > 1, "Expected multiple wells"

    def test_multiple_channels(self, imagexpress_dir, requires_data):
        if imagexpress_dir is None:
            pytest.skip("No imagexpress/ subdir in test data")
        result = load_plate(imagexpress_dir)
        plate = result["plate"]
        well = next(iter(plate.wells.values()))
        site = next(iter(well.sites.values()))
        images = site.get_images()
        arr = next(iter(images.values()))
        # Expect at least a channel dimension > 1
        assert arr.ndim >= 3


# ---------------------------------------------------------------------------
# Phenix
# ---------------------------------------------------------------------------

class TestPhenixIntegration:
    def test_load_plate(self, phenix_dir, requires_data):
        if phenix_dir is None:
            pytest.skip("No phenix/ subdir in test data")
        result = load_plate(phenix_dir)
        _assert_valid_plate(result)

    def test_well_naming(self, phenix_dir, requires_data):
        if phenix_dir is None:
            pytest.skip("No phenix/ subdir in test data")
        result = load_plate(phenix_dir)
        for well_name in result["plate"].wells:
            # Phenix wells are named r##c## (e.g. r01c02)
            assert well_name.startswith("r"), (
                f"Unexpected well name format: {well_name}"
            )


# ---------------------------------------------------------------------------
# CZI (Zeiss CellDiscoverer 7)
# ---------------------------------------------------------------------------

class TestCziIntegration:
    def test_load_plate(self, czi_dir, requires_data):
        if czi_dir is None:
            pytest.skip("No czi/ subdir in test data")
        result = load_plate(czi_dir)
        _assert_valid_plate(result)

    def test_well_naming(self, czi_dir, requires_data):
        if czi_dir is None:
            pytest.skip("No czi/ subdir in test data")
        result = load_plate(czi_dir)
        for well_name in result["plate"].wells:
            # CZI wells are named row##col## (e.g. row1col2)
            assert well_name.startswith("row"), (
                f"Unexpected well name format: {well_name}"
            )


# ---------------------------------------------------------------------------
# SiteTiff
# ---------------------------------------------------------------------------

class TestSiteTiffIntegration:
    def test_load_plate(self, sitetiff_dir, requires_data):
        if sitetiff_dir is None:
            pytest.skip("No sitetiff/ subdir in test data")
        result = load_plate(sitetiff_dir)
        _assert_valid_plate(result)

    def test_load_labels(self, sitetiff_dir, sitetiff_labels_dir, requires_data):
        if sitetiff_dir is None or sitetiff_labels_dir is None:
            pytest.skip("No sitetiff/ or sitetiff_labels/ subdir in test data")
        # Load images first (populates plate in state)
        load_plate(sitetiff_dir)
        # Then load labels on top
        result = load_plate(
            sitetiff_labels_dir, iol="label", name="segmentation"
        )
        plate = result["plate"]
        well = next(iter(plate.wells.values()))
        site = next(iter(well.sites.values()))
        labels = site.get_labels()
        assert "segmentation" in labels
        arr = labels["segmentation"]
        assert isinstance(arr, da.Array)
