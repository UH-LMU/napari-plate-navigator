"""
Integration tests for napari-plate-navigator.

These tests require real microscopy data. They are automatically skipped
if no data directory is provided. See conftest.py for how to supply data.

Run with:
    pytest --data-dir /path/to/testdata
    TEST_DATA_DIR=/path/to/testdata pytest
    TEST_DATA_URL=https://a3s.fi/bucket/testdata pytest

Or per-format:
    pytest --imagexpress-dir ./data/imagexpress --phenix-dir /path/to/phenix
"""

import dask.array as da
import pytest

try:
    import xarray as xr

    _has_xarray = True
except ImportError:
    _has_xarray = False

from napari_plate_navigator._reader import load_plate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_none(path, label):
    if path is None:
        pytest.skip(f"No {label} data provided")


def _is_array_like(arr) -> bool:
    """True for dask arrays and xarray DataArrays (which wrap dask arrays)."""
    return isinstance(arr, da.Array) or (
        _has_xarray and isinstance(arr, xr.DataArray)
    )


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
                assert _is_array_like(arr), (
                    f"Expected dask/xarray for {well_name}/{site_name}/{name}, "
                    f"got {type(arr)}"
                )
            return  # One site is enough for a smoke test


# ---------------------------------------------------------------------------
# ImageXpress
# ---------------------------------------------------------------------------


class TestImageXpressIntegration:
    def test_load_plate(self, imagexpress_dir):
        _skip_if_none(imagexpress_dir, "imagexpress")
        result = load_plate(imagexpress_dir)
        _assert_valid_plate(result)

    def test_multiple_wells(self, imagexpress_dir):
        _skip_if_none(imagexpress_dir, "imagexpress")
        result = load_plate(imagexpress_dir)
        assert result["plate"].nwells() > 1, "Expected multiple wells"

    def test_multiple_channels(self, imagexpress_dir):
        _skip_if_none(imagexpress_dir, "imagexpress")
        result = load_plate(imagexpress_dir)
        plate = result["plate"]
        well = next(iter(plate.wells.values()))
        site = next(iter(well.sites.values()))
        images = site.get_images()
        arr = next(iter(images.values()))
        assert arr.ndim >= 3


# ---------------------------------------------------------------------------
# Phenix
# ---------------------------------------------------------------------------


class TestPhenixIntegration:
    def test_load_plate(self, phenix_dir):
        _skip_if_none(phenix_dir, "phenix")
        result = load_plate(phenix_dir)
        _assert_valid_plate(result)

    def test_well_naming(self, phenix_dir):
        _skip_if_none(phenix_dir, "phenix")
        result = load_plate(phenix_dir)
        for well_name in result["plate"].wells:
            assert well_name.startswith(
                "r"
            ), f"Unexpected well name format: {well_name}"

    def test_load_labels(self, phenix_dir, phenix_labels_dir):
        _skip_if_none(phenix_dir, "phenix images")
        _skip_if_none(phenix_labels_dir, "phenix labels")
        load_plate(phenix_dir)
        result = load_plate(phenix_labels_dir, iol="label", name="cellpose")
        plate = result["plate"]
        well = next(iter(plate.wells.values()))
        site = next(iter(well.sites.values()))
        labels = site.get_labels()
        assert "cellpose" in labels
        assert isinstance(labels["cellpose"], da.Array)


# ---------------------------------------------------------------------------
# CZI (Zeiss CellDiscoverer 7)
# ---------------------------------------------------------------------------


class TestCziIntegration:
    def test_load_plate(self, czi_dir):
        _skip_if_none(czi_dir, "czi")
        result = load_plate(czi_dir)
        _assert_valid_plate(result)

    def test_well_naming(self, czi_dir):
        _skip_if_none(czi_dir, "czi")
        result = load_plate(czi_dir)
        for well_name in result["plate"].wells:
            assert well_name.startswith(
                "row"
            ), f"Unexpected well name format: {well_name}"


# ---------------------------------------------------------------------------
# SiteTiff
# ---------------------------------------------------------------------------


class TestSiteTiffIntegration:
    def test_load_plate(self, sitetiff_dir):
        _skip_if_none(sitetiff_dir, "sitetiff")
        result = load_plate(sitetiff_dir)
        _assert_valid_plate(result)

    def test_load_labels(self, sitetiff_dir, sitetiff_labels_dir):
        _skip_if_none(sitetiff_dir, "sitetiff images")
        _skip_if_none(sitetiff_labels_dir, "sitetiff labels")
        load_plate(sitetiff_dir)
        result = load_plate(
            sitetiff_labels_dir, iol="label", name="segmentation"
        )
        plate = result["plate"]
        well = next(iter(plate.wells.values()))
        site = next(iter(well.sites.values()))
        labels = site.get_labels()
        assert "segmentation" in labels
        assert isinstance(labels["segmentation"], da.Array)
