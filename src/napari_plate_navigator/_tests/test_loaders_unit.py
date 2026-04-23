"""
Unit tests for loader classes.

These tests require no real microscopy data — they use empty files with
instrument-appropriate naming patterns (created in tmp_path) to verify
filename parsing and format detection logic.
"""

import pandas as pd

from napari_plate_navigator._base import Plate
from napari_plate_navigator._reader import (
    CHANNEL,
    PLATE,
    SITE,
    TSTEP,
    WELL,
    ZSTEP,
    CziLoader,
    ImageXpressLoader,
    PhenixLoader,
    SiteTiffLabelLoader,
    SiteTiffLoader,
    build_plate_from_df_fast,
)

# ---------------------------------------------------------------------------
# ImageXpressLoader
# ---------------------------------------------------------------------------


class TestImageXpressLoader:
    def test_can_read_valid(self, ix_files):
        loader = ImageXpressLoader()
        assert loader.can_read(ix_files[0])

    def test_can_read_rejects_phenix(self, phenix_files):
        loader = ImageXpressLoader()
        assert not loader.can_read(phenix_files[0])

    def test_can_read_rejects_sitetiff(self, sitetiff_files):
        loader = ImageXpressLoader()
        assert not loader.can_read(sitetiff_files[0])

    def test_discover_metadata_columns(self, ix_files):
        loader = ImageXpressLoader()
        df = loader.discover_metadata(ix_files)
        for col in [PLATE, WELL, SITE, CHANNEL, TSTEP, ZSTEP]:
            assert col in df.columns, f"Missing column: {col}"

    def test_discover_metadata_wells(self, ix_files):
        loader = ImageXpressLoader()
        df = loader.discover_metadata(ix_files)
        wells = set(df[WELL].unique())
        assert "A01" in wells
        assert "B02" in wells

    def test_discover_metadata_sites(self, ix_files):
        loader = ImageXpressLoader()
        df = loader.discover_metadata(ix_files)
        assert set(df[df[WELL] == "A01"][SITE].unique()) == {1, 2}

    def test_discover_metadata_channels(self, ix_files):
        loader = ImageXpressLoader()
        df = loader.discover_metadata(ix_files)
        # A01/s1 has channels 1 and 2
        a01_s1 = df[(df[WELL] == "A01") & (df[SITE] == 1)]
        assert set(a01_s1[CHANNEL].unique()) == {1, 2}

    def test_discover_metadata_excludes_thumbs(self, tmp_path):
        plate = tmp_path / "plate1"
        plate.mkdir()
        ok = plate / "t001_A01_s1_w1_z001.tif"
        thumb = plate / "t001_A01_s1_w1_z001_thumb.tif"
        ok.touch()
        thumb.touch()
        loader = ImageXpressLoader()
        df = loader.discover_metadata([ok, thumb])
        assert len(df) == 1


# ---------------------------------------------------------------------------
# PhenixLoader
# ---------------------------------------------------------------------------


class TestPhenixLoader:
    def test_can_read_valid(self, phenix_files):
        loader = PhenixLoader()
        assert loader.can_read(phenix_files[0])

    def test_can_read_rejects_ix(self, ix_files):
        loader = PhenixLoader()
        assert not loader.can_read(ix_files[0])

    def test_can_read_rejects_sitetiff(self, sitetiff_files):
        loader = PhenixLoader()
        assert not loader.can_read(sitetiff_files[0])

    def test_discover_metadata_columns(self, phenix_files):
        loader = PhenixLoader()
        df = loader.discover_metadata(phenix_files)
        for col in [PLATE, WELL, SITE, CHANNEL, TSTEP, ZSTEP]:
            assert col in df.columns, f"Missing column: {col}"

    def test_discover_metadata_wells(self, phenix_files):
        loader = PhenixLoader()
        df = loader.discover_metadata(phenix_files)
        wells = set(df[WELL].unique())
        assert "r01c01" in wells
        assert "r02c03" in wells

    def test_discover_metadata_channels(self, phenix_files):
        loader = PhenixLoader()
        df = loader.discover_metadata(phenix_files)
        r01c01 = df[df[WELL] == "r01c01"]
        assert set(r01c01[CHANNEL].unique()) == {1, 2}

    def test_discover_metadata_sorted(self, phenix_files):
        loader = PhenixLoader()
        df = loader.discover_metadata(phenix_files)
        # Should be sorted by [PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL]
        sorted_df = df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sorted_df)


# ---------------------------------------------------------------------------
# SiteTiffLoader
# ---------------------------------------------------------------------------


class TestSiteTiffLoader:
    def test_can_read_valid(self, sitetiff_files):
        loader = SiteTiffLoader()
        assert loader.can_read(sitetiff_files[0])

    def test_can_read_rejects_ix(self, ix_files):
        loader = SiteTiffLoader()
        assert not loader.can_read(ix_files[0])

    def test_can_read_rejects_phenix(self, phenix_files):
        loader = SiteTiffLoader()
        assert not loader.can_read(phenix_files[0])

    def test_discover_metadata_columns(self, sitetiff_files):
        loader = SiteTiffLoader()
        df = loader.discover_metadata(sitetiff_files)
        for col in [PLATE, WELL, SITE, CHANNEL, TSTEP, ZSTEP]:
            assert col in df.columns, f"Missing column: {col}"

    def test_discover_metadata_wells(self, sitetiff_files):
        loader = SiteTiffLoader()
        df = loader.discover_metadata(sitetiff_files)
        wells = set(df[WELL].unique())
        assert "row1col1" in wells
        assert "row2col3" in wells

    def test_discover_metadata_sites(self, sitetiff_files):
        loader = SiteTiffLoader()
        df = loader.discover_metadata(sitetiff_files)
        row1col1 = df[df[WELL] == "row1col1"]
        assert set(row1col1[SITE].unique()) == {1, 2}


# ---------------------------------------------------------------------------
# SiteTiffLabelLoader
# ---------------------------------------------------------------------------


class TestSiteTiffLabelLoader:
    def test_can_read_valid(self, sitetiff_files):
        # Inherits can_read from SiteTiffLoader
        loader = SiteTiffLabelLoader()
        assert loader.can_read(sitetiff_files[0])


# ---------------------------------------------------------------------------
# CziLoader
# ---------------------------------------------------------------------------


class TestCziLoader:
    def test_can_read_valid(self, czi_file):
        loader = CziLoader()
        assert loader.can_read(czi_file)

    def test_can_read_rejects_tif(self, ix_files):
        loader = CziLoader()
        assert not loader.can_read(ix_files[0])

    def test_can_read_case_insensitive(self, tmp_path):
        f = tmp_path / "experiment.CZI"
        f.touch()
        loader = CziLoader()
        assert loader.can_read(f)


# ---------------------------------------------------------------------------
# build_plate_from_df_fast
# ---------------------------------------------------------------------------


class TestBuildPlate:
    def _make_df(self):
        return pd.DataFrame(
            {
                "Path": ["/data/img1.tif", "/data/img2.tif"],
                PLATE: ["plate1", "plate1"],
                WELL: ["A01", "A01"],
                SITE: [1, 2],
                CHANNEL: [1, 1],
                TSTEP: [1, 1],
                ZSTEP: [1, 1],
            }
        )

    def test_builds_wells(self):
        plate = Plate()
        build_plate_from_df_fast(
            self._make_df(), plate, iol="image", name="img"
        )
        assert set(plate.wells.keys()) == {"A01"}

    def test_builds_sites(self):
        plate = Plate()
        build_plate_from_df_fast(
            self._make_df(), plate, iol="image", name="img"
        )
        well = plate.get_well("A01")
        assert set(well.sites.keys()) == {"1", "2"}

    def test_stores_filelist(self):
        plate = Plate()
        build_plate_from_df_fast(
            self._make_df(), plate, iol="image", name="img"
        )
        site = plate.get_well_site("A01", "1")
        assert "img" in site.filelists_img
        assert len(site.filelists_img["img"]) == 1

    def test_label_iol(self):
        plate = Plate()
        build_plate_from_df_fast(
            self._make_df(), plate, iol="label", name="nuclei"
        )
        site = plate.get_well_site("A01", "1")
        assert "nuclei" in site.filelists_lbl
