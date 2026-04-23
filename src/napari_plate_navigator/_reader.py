# src/napari_plate_navigator/_reader.py
import logging
import os
import re
from pathlib import Path
from typing import Any

import bioio_czi  # For explicit CZI support
import dask.array as da
import pandas as pd
from bioio import BioImage
from czitools.metadata_tools.czi_metadata import CziMetadata
from tqdm import tqdm

from ._base import Plate, StateManager
from ._utils import log_method

logger = logging.getLogger(__name__)  # Module-level logger

# Constants
DIR = "Directory"
PATH = "Path"
PLATE = "Plate"
WELL = "Well"
SITE = "Site"
CHANNEL = "Channel"
TSTEP = "TStep"
ZSTEP = "ZStep"


class BaseLoader:
    """Abstract base for file loaders."""

    def can_read(self, path: Path) -> bool:
        """Return True if loader can read the given file."""
        raise NotImplementedError

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Extract T/Z/C/WELL/SITE/PLATE from files/metadata, return df."""
        raise NotImplementedError

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        """Lazy-load a slice as dask array."""
        raise NotImplementedError

    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Loader-specific: Stack group_df to per-site (T,Z,C,Y,X) array."""
        raise NotImplementedError


class TiffLoader(BaseLoader):
    def can_read(self, path: Path) -> bool:
        try:
            metadata = self.discover_metadata([path])
            if WELL in metadata.columns:
                return True
        except ValueError as e:
            logger.debug(e.__class__.__name__)
            logger.warning(
                "%s failed to read metadata: %s", self.__class__.__name__, e
            )

        return False

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        img = BioImage(str(path))
        # Lazy slice via BioImage params (works for embedded dims or single)
        scene_kwargs = {}
        if t is not None:
            scene_kwargs["T"] = t
        if z is not None:
            scene_kwargs["Z"] = z
        if c is not None:
            scene_kwargs["C"] = c
        data = img.get_image_dask_data(**scene_kwargs).squeeze()
        return data

    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Multi-file: Loop T/Z/C paths, stack slices."""
        exploded = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])
        t_steps = []
        for _tstep, t_group in exploded.groupby(TSTEP):
            z_steps = []
            for _zstep, z_group in t_group.groupby(ZSTEP):
                logger.debug(
                    "z_group: %s",
                    [Path(p).name for p in z_group[PATH].values],
                )
                channels = [
                    self.load_slice(Path(p)) for p in z_group[PATH]
                ]  # Full per-file (C=1)
                try:
                    z_stack = da.stack(channels, axis=0)  # Stack C
                except ValueError as ve:
                    logger.error("Failed to stack channels: %s", ve)
                    logger.error("paths: %s", site_group[PATH].values)
                    raise
                z_steps.append(z_stack)
            try:
                t_stack = da.stack(z_steps, axis=0)  # Stack Z
            except ValueError as ve:
                logger.error("Failed to stack zsteps: %s", ve)
                logger.error("paths: %s", site_group[PATH].values)
                raise
            t_steps.append(t_stack)
        return da.stack(t_steps, axis=0)  # Stack T


class ImageXpressLoader(TiffLoader):
    """Specific loader for Molecular Devices ImageXpress (regex + proj logic)."""

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        files = [f for f in files if "thumb" not in f.name]
        df = pd.DataFrame({PATH: [str(f) for f in files]})
        metadata_columns = {
            "mc2": TSTEP,
            "mc3": ZSTEP,
            "mc4": PLATE,
            "mc5": WELL,
            "mc6": SITE,
            "mc7": CHANNEL,
        }
        pattern = (
            r"[/\\](?P<{mc4}>[^/\\]*)"
            r"(?:[/\\][^/\\]*_Projection)?"
            r"(?:[/\\]timepoint\d+)?"
            r"[/\\]t(?P<{mc2}>\d+)_(?P<{mc5}>\w\d{{2}})_s(?P<{mc6}>\d{{1,2}})_w(?P<{mc7}>\d)_z(?P<{mc3}>\d+)"
        ).format(**metadata_columns)
        extracted = df[PATH].str.extract(pattern)
        df = df.join(extracted)

        try:
            df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
            df[PLATE] = df[PLATE].astype(str)
            df[WELL] = df[WELL].astype(str)
            df[SITE] = df[SITE].astype(int)
            df[CHANNEL] = df[CHANNEL].astype(int)
            df[TSTEP] = df[TSTEP].astype(int)
            df[ZSTEP] = df[ZSTEP].astype(int)
        except ValueError as e:
            logger.error("Failed to convert metadata type %s", e)
            df.to_csv(
                f"/home/{os.getenv('USER')}/tmp/imageXpress_metadata.csv"
            )
            raise

        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        return df

    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Multi-file: Loop T outer, C middle, Z inner. Repeat single-Z channels to full_Z."""
        exploded = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])
        # explode() converts int columns to object dtype when values are scalars
        for col in [TSTEP, ZSTEP, CHANNEL]:
            exploded[col] = pd.to_numeric(exploded[col])

        # Number of Z slices in stack channels (for repeating projections to match).
        # Count unique values rather than max+1 to handle 1-indexed z-steps.
        full_z = int(exploded[exploded[ZSTEP] > 0][ZSTEP].nunique())
        logger.debug("full_z %d", full_z)

        t_steps = []
        for _tstep, t_group in exploded.groupby(TSTEP):
            c_groups = []
            for channel, c_group in t_group.groupby(CHANNEL):
                logger.debug("channel %s", channel)
                z_steps = []
                for _zstep, z_subgroup in c_group.groupby(ZSTEP):
                    # Assume one path per Z/C/T (HCS norm)
                    path = z_subgroup[PATH].iloc[0]
                    logger.debug("path %s", path)
                    data = self.load_slice(
                        Path(path)
                    )  # Full per-file (Y,X or small)
                    z_steps.append(data)
                # Stack Z for this C/T
                c_stack = da.stack(z_steps, axis=0)  # Shape (Z_actual, Y, X)
                logger.debug("c_stack.shape %s", c_stack.shape)

                # Repeat if single-Z (proj/single channels)
                if c_stack.shape[0] == 1 and (full_z > 2):
                    c_stack = da.repeat(
                        c_stack, repeats=full_z, axis=0
                    )  # (full_Z, Y, X)
                elif c_stack.shape[0] != full_z:
                    logger.warning(
                        "Channel %s has %s Z != full %s; truncating",
                        channel,
                        c_stack.shape[0],
                        full_z,
                    )
                    c_stack = c_stack[:full_z]  # Or pad if < full
                logger.debug("c_stack.shape %s", c_stack.shape)

                c_groups.append(c_stack)

            # Stack C for this T
            t_c_stack = da.stack(
                c_groups, axis=-1
            )  # (full_Z, Y, X, C) → transpose to (C, full_Z, Y, X) if needed
            t_c_stack = da.moveaxis(t_c_stack, -1, 0)  # To (C, full_Z, Y, X)
            t_steps.append(t_c_stack)

        # Stack T: (T, C, full_Z, Y, X) → transpose to (T, full_Z, C, Y, X) for Napari
        full_array = da.stack(t_steps, axis=0)
        full_array = da.moveaxis(
            full_array, 1, 2
        )  # (T, C, full_Z, Y, X) → (T, full_Z, C, Y, X)
        return full_array


class PhenixLoader(TiffLoader):

    def __init__(self):
        metadata_columns = {
            "mc1": WELL,
            "mc2": SITE,
            "mc3": ZSTEP,
            "mc4": CHANNEL,
            "mc5": PLATE,
        }
        self.file_pattern = (
            r"(?P<{mc1}>r\d\dc\d\d)f(?P<{mc2}>\d\d)p(?P<{mc3}>\d\d)"
            r"-ch(?P<{mc4}>\d)"
        ).format(**metadata_columns)

    def can_read(self, path: Path) -> bool:
        logger.debug(self.file_pattern)
        logger.debug(str(path))
        return re.search(self.file_pattern, path.name) is not None

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        df = pd.DataFrame({PATH: [str(f) for f in files]})
        df.drop_duplicates(inplace=True)
        logger.debug("phxldr: df.shape %s", df.shape)
        extracted = df[PATH].str.extract(self.file_pattern)
        logger.debug("phxldr: extracted.shape %s", extracted.shape)
        df = df.join(extracted)
        logger.debug("phxldr: df.shape %s", df.shape)

        df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
        df[PLATE] = "plate"
        df[WELL] = df[WELL].astype(str)

        for col in [CHANNEL, SITE, ZSTEP]:
            # remove first leading zero? apparently not needed.
            df[col] = df[col].str.replace(r"^0", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # fix timestep for now
        df[TSTEP] = 1

        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )

        return df


def build_well_df(row, col, idx):
    """
    Build a DataFrame from row/col/site arrays, with 'Well' derived for grouping.

    Site       — 0-based index within each well (matches segment-czi output filenames).
    SceneIndex — original CZI scene index from metadata (used for CZI reads).
    """
    df = pd.DataFrame({"Row": row, "Col": col, "SceneIndex": idx})
    df["Well"] = "row" + df["Row"].astype(str) + "col" + df["Col"].astype(str)
    df["Site"] = df.groupby("Well").cumcount()
    return df


class SiteTiffLoader(BaseLoader):
    def can_read(self, path: Path) -> bool:
        filepattern = r"well_(.*)_site_(\d\d\d).*tif$"
        return re.search(filepattern, path.name) is not None

    @log_method
    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        logger.debug(files[0].name)
        df = pd.DataFrame({PATH: [str(f) for f in files]})
        # Extract from filename only (not full path)
        df["filename"] = df[PATH].apply(lambda x: Path(x).name)
        metadata_columns = {
            "mc1": WELL,
            "mc2": SITE,
        }
        pattern = (
            r"well_(?P<{mc1}>row\d*col\d*)_site_(?P<{mc2}>\d{{3}})_.*\.tif"
        ).format(**metadata_columns)
        extracted = df["filename"].str.extract(pattern)
        df = df.drop(columns=["filename"]).join(extracted)
        logger.debug(df)

        df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
        df[WELL] = df[WELL].astype(str)
        df[SITE] = df[SITE].astype(int)
        df[PLATE] = "dummy_plate_name"
        df[CHANNEL] = 0
        df[TSTEP] = 0
        df[ZSTEP] = 0

        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        return df

    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Single file: whole site in one .tiff"""
        filename = site_group[PATH].values[0]
        logger.debug(filename)
        img = BioImage(filename)
        logger.debug("img.dims %s", img.dims)
        logger.debug("img.shape %s", img.shape)
        logger.debug("img.channel_names %s", img.channel_names)
        logger.debug("img.ome_metadata %s", img.ome_metadata)

        # dd = img.get_image_dask_data()
        xr = img.get_xarray_dask_stack()  # .isel(I=0)
        logger.debug("xr.dims %s", xr.dims)
        logger.debug("xr.shape %s", xr.shape)

        return xr


class SiteTiffLabelLoader(SiteTiffLoader):
    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Single file: whole site in one .tiff"""
        filename = site_group[PATH].values[0]
        logger.debug(filename)
        img = BioImage(filename)
        logger.debug("img.dims %s", img.dims)
        logger.debug("img.shape %s", img.shape)
        logger.debug("img.channel_names %s", img.channel_names)
        logger.debug("img.ome_metadata %s", img.ome_metadata)

        # dd = img.get_image_dask_data()
        xr = img.get_xarray_dask_stack()  # .isel(I=0)
        logger.debug("xr.dims %s", xr.dims)
        logger.debug("xr.shape %s", xr.shape)

        # drop C dimension to make label work
        xr = xr.isel(C=0, drop=True)
        logger.debug("xr.dims %s", xr.dims)
        logger.debug("xr.shape %s", xr.shape)

        return xr


class CziLoader(BaseLoader):
    """Loader for Zeiss CZI files (monolithic or multi-scene)."""

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".czi"

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Parse filename for WELL/SITE/PLATE, embedded for T/Z/C."""
        if not files:
            return pd.DataFrame()

        # exclude thumbnail images
        files = [f for f in files if f.suffix.lower() == ".czi"]

        path = files[0]  # Assume single file for now; extend for multi later

        # reconstructing mosaic will take ages, skip
        img = BioImage(
            str(path), reconstruct_mosaic=True, reader=bioio_czi.Reader
        )

        # store xarray in state
        StateManager.get_instance().czi = img

        # TODO: use czitools to read metadata
        md = CziMetadata(str(path))
        row_id = md.sample.well_rowID
        col_id = md.sample.well_colID
        scene_id = md.sample.well_indices
        assert len(row_id) == len(col_id)
        assert len(col_id) == len(scene_id)

        # construct dataframe the holds the well/site combinations
        df = build_well_df(row_id, col_id, scene_id)

        # fill in other columns with dummy values
        df[PATH] = str(path)
        df[PLATE] = "dummy_plate_name"
        df[CHANNEL] = 0
        df[TSTEP] = 0
        df[ZSTEP] = 0

        return df

    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        logger.debug(site_group[[WELL, SITE]])
        # SceneIndex is the original CZI scene index; Site is per-well 0-based
        scene_index = int(site_group["SceneIndex"].values[0])
        logger.debug("scene_index %s", scene_index)

        # use img stored in state
        img = StateManager.get_instance().czi
        xr = img.get_xarray_dask_stack(select_scenes=(scene_index,))
        logger.debug("xr.dims %s", xr.dims)
        logger.debug("xr.shape %s", xr.shape)

        return xr


@log_method
def get_loader(path: Path) -> BaseLoader:
    loaders = [
        CziLoader(),
        ImageXpressLoader(),
        SiteTiffLoader(),
        SiteTiffLabelLoader(),
        PhenixLoader(),
    ]
    for loader in loaders:
        if loader.can_read(path):
            logger.debug("selected %s", str(loader))
            return loader

    raise ValueError("No loader found for " + str(path))


@log_method
def load_plate(
    directory: Path,
    wells: list[str] | None = None,
    nwells: int = -1,
    nsites: int = -1,
    iol: str = "image",  # or 'label'
    name: str = "image_or_label_name",
) -> dict[str, Any]:
    """Load a folder of microscopy files into a Plate.

    Returns: {'df': pd.DataFrame, 'plate': Plate}
    """
    if not directory.exists():
        logger.error("Directory does not exist: %s", directory)
        return {"df": pd.DataFrame(), "plate": Plate()}

    # Glob all allowed files
    formats = [".czi", ".png", ".tif", ".tiff"]

    files = []

    for root, _dirs, filenames in os.walk(directory, followlinks=True):
        root_path = Path(root)
        for filename in filenames:
            p = root_path / filename
            if p.suffix.lower() in formats:
                files.append(p)

    if not files:
        logger.error("No files found in directory: %s", directory)
        return {"df": pd.DataFrame(), "plate": Plate()}

    logger.debug("len(files) %d", len(files))

    loader = get_loader(files[0])
    if iol == "label" and isinstance(loader, SiteTiffLoader):
        loader = SiteTiffLabelLoader()

    # Discover metadata/df
    df = loader.discover_metadata(files)
    if df.empty:
        return {"df": df, "plate": Plate()}

    # Post-process df (sorting, filtering—shared across loaders)
    df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
    df = df.astype(
        {
            PLATE: str,
            WELL: str,
            SITE: int,
            CHANNEL: int,
            TSTEP: int,
            ZSTEP: int,
        }
    )
    df.sort_values(
        by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
        inplace=True,
        ignore_index=True,
    )

    if wells:
        df = df[df[WELL].isin(wells)]
    elif nwells > 0:
        unique_wells = df[WELL].unique()[:nwells]
        df = df[df[WELL].isin(unique_wells)]
    if nsites > 0:
        df = df[df[SITE] <= nsites]

    state = StateManager.get_instance()
    if iol == "image":
        state.clear_state()
        state.plate = Plate()
        state.df_images = df
        state.loader_img = loader
    elif iol == "label":
        state.loader_lbl = loader

    logger.debug("name %s", name)

    plate = StateManager.get_instance().plate
    build_plate_from_df_fast(df, plate, iol=iol, name=name)

    return {"df": df, "plate": plate}


@log_method
def build_plate_from_df_fast(
    stacks_df: pd.DataFrame,
    plate: Plate,
    iol: str = "image",
    name: str = "image_name",
):
    grouped_df = stacks_df.groupby([WELL, SITE])

    for (well, site), site_group in tqdm(grouped_df, desc="Prepping sites"):
        logger.debug("well %s site %s", well, site)
        well_site = plate.get_well_site(well, site)
        if iol == "image":
            well_site.set_filelist_img(name, site_group)
        elif iol == "label":
            well_site.set_filelist_lbl(name, site_group)
