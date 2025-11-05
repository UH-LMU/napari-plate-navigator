# src/napari_plate_navigator/_reader.py
import logging
import os
from collections import defaultdict
from pathlib import Path
import re
from typing import Any

import dask.array as da
import pandas as pd
from bioio import BioImage
import bioio_czi # For explicit CZI support
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

    def separate_stacks_and_aux(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        return df, {}  # Generic: No aux

    def get_extra_metadata(self, path: Path) -> dict[str, Any]:
        return {}  # Generic


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

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Simple df from paths; no parsing—fallback for non-structured files."""
        df = pd.DataFrame({"Path": [str(f) for f in files]})
        df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
        # Assign defaults (extend with basic regex if needed)
        df[PLATE] = "1"
        df[WELL] = "A1"
        df[SITE] = 1
        df[TSTEP] = 0
        df[ZSTEP] = 0
        df[CHANNEL] = "0"
        return df.astype(
            {
                PLATE: str,
                WELL: str,
                SITE: int,
                TSTEP: int,
                ZSTEP: int,
                CHANNEL: str,
            }
        )

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
                channels = [
                    self.load_slice(Path(p)) for p in z_group[PATH]
                ]  # Full per-file (C=1)
                z_stack = da.stack(channels, axis=0)  # Stack C
                z_steps.append(z_stack)
            t_stack = da.stack(z_steps, axis=0)  # Stack Z
            t_steps.append(t_stack)
        return da.stack(t_steps, axis=0)  # Stack T


class ImageXpressLoader(TiffLoader):
    """Specific loader for Molecular Devices ImageXpress (regex + proj logic)."""

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        # exclude thumbnail images
        files = [f for f in files if "thumb" not in f.name]

        """Use original regex for MolDev naming."""
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

    # In _reader.py (Updated ImageXLoader.build_site_array)
    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        """Multi-file: Loop T outer, C middle, Z inner. Repeat single-Z channels to full_Z."""
        exploded = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])

        # Find global max Z from channels with stacks (for repetition)
        full_z = (
            exploded[exploded[ZSTEP] > 0][ZSTEP].max() + 1
        )  # e.g., 10 Z slices
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

    @log_method
    def separate_stacks_and_aux(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        # Your get_stacks_and_projections code here
        ch_stack = df[df[ZSTEP] > 1][CHANNEL].unique()
        ch_proj = df[df[ZSTEP] == 0][CHANNEL].unique()
        logger.info("ch_stack: %s", ch_stack)
        logger.info("ch_proj: %s", ch_proj)

        # list of unique channel/zstep combinations
        cz = df[[CHANNEL, ZSTEP]].drop_duplicates()
        # remove projections
        cz = cz[cz[ZSTEP] != 0]
        gcz = cz.groupby(CHANNEL).count().reset_index()
        print(gcz)
        mask = gcz["ZStep"] != 1
        ch_single = gcz[mask][CHANNEL].values
        logger.info("ch_single: %s", ch_proj)

        stacks = pd.DataFrame()
        projs = pd.DataFrame()
        singles = pd.DataFrame()

        # ignore duplicates in _Projection/
        df = df[~df[DIR].str.endswith("_Projection")]

        projs = df[df[CHANNEL].isin(ch_proj)].copy().reset_index(drop=True)
        aux = {"projection": projs}  # Aux dict
        if len(ch_stack) > 0:
            stacks = (
                df[df[CHANNEL].isin(ch_stack)].copy().reset_index(drop=True)
            )
            singles = (
                df[df[CHANNEL].isin(ch_single)].copy().reset_index(drop=True)
            )
            aux["single_image"] = singles
        else:
            stacks = df[df[ZSTEP] == 1].copy().reset_index(drop=True)

        # stacks.sort_values(
        #    by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
        #    inplace=True,
        #    ignore_index=True,
        # )

        return stacks, aux


class PhenixLoader(TiffLoader):

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        df = pd.DataFrame({PATH: [str(f) for f in files]})
        metadata_columns = {
            "mc1": WELL,
            "mc2": SITE,
            "mc3": ZSTEP,
            "mc4": CHANNEL,
            "mc5": PLATE,
        }
        pattern = (
            r"[/\\](?P<{mc5}>[^/\\]*)"
            r"[/\\](?P<{mc1}>r\d\dc\d\d)f(?P<{mc2}>\d\d)p(?P<{mc3}>\d\d)"
            r"-ch(?P<{mc4}>\d)"
        ).format(**metadata_columns)
        extracted = df[PATH].str.extract(pattern)
        df = df.join(extracted)

        # remove leading zeros
        for col in [SITE, CHANNEL, ZSTEP]:
            df[col] = df[col].str.lstrip("0")

        df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
        df[PLATE] = df[PLATE].astype(str)
        df[WELL] = df[WELL].astype(str)
        df[SITE] = df[SITE].astype(int)
        df[CHANNEL] = df[CHANNEL].astype(int)
        # df[TSTEP] = df[TSTEP].astype(int)
        df[ZSTEP] = df[ZSTEP].astype(int)

        # fix timestep for now
        df[TSTEP] = 1

        # df.to_csv(f"/home/{user}/tmp/PhenixLoader.get_metadata.csv")

        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        return df


def build_well_df(row, col, idx):
    """
    Build a DataFrame from row/col/site arrays, with 'Well' derived for grouping.
    
    Args:
        row: List-like of row indices (e.g., [1, 1, 2, 2])
        col: List-like of col indices (e.g., [1, 2, 1, 2])
        idx: List-like of site indices (e.g., [1, 2, 3, 4])
    
    Returns:
        pd.DataFrame with columns ['Row', 'Col', 'Site', 'Well']
    """
    df = pd.DataFrame({
        'Row': row,
        'Col': col,
        'Site': idx
    })
    df['Well'] = 'row' + df['Row'].astype(str) + 'col' + df['Col'].astype(str)
    return df


class SiteTiffLoader(BaseLoader):
    def can_read(self, path: Path) -> bool:
        filepattern = r"well_(.*)_site_(\d\d\d).*tif$"
        return re.search(filepattern, path.name) != None

    @log_method
    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        logger.debug(files[0].name)
        df = pd.DataFrame({PATH: [str(f) for f in files]})
        # Extract from filename only (not full path)
        df['filename'] = df[PATH].apply(lambda x: Path(x).name)
        metadata_columns = {
            "mc1": WELL,
            "mc2": SITE,
        }
        pattern = (
            r"well_(?P<{mc1}>row\d*col\d*)_site_(?P<{mc2}>\d{{3}})_.*\.tif"
        ).format(**metadata_columns)
        extracted = df['filename'].str.extract(pattern)
        #df = df.drop(columns=['filename']).join(extracted)  # Drop temp column
        df = df.join(extracted)  # Drop temp column
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

        #dd = img.get_image_dask_data()
        xr = img.get_xarray_dask_stack()#.isel(I=0)
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
        img = BioImage(str(path),
                       reconstruct_mosaic=False,
                       reader=bioio_czi.Reader)

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
        df[PLATE] = 'dummy_plate_name'
        df[CHANNEL] = 0
        df[TSTEP] = 0
        df[ZSTEP] = 0
        
        return df

    @log_method
    def build_site_array(self, site_group: pd.DataFrame) -> da.Array:
        logger.debug(site_group[[WELL,SITE]])
        # convert to int for get_xarray_dask_stack()
        site = int(site_group[SITE].values[0])
        logger.debug("site %s", site)
        
        # use img stored in state
        img = StateManager.get_instance().czi
        xr = img.get_xarray_dask_stack(select_scenes=(site,))
        logger.debug("xr.dims %s", xr.dims)
        logger.debug("xr.shape %s", xr.shape)

        return xr
        
    def get_extra_metadata(self, path: Path) -> dict[str, Any]:
        img = BioImage(str(path), reader=bioio_czi.Reader)
        # Extract your example fields (extend as needed)
        metadata = {
            "filename": path.name,
            "extension": path.suffix,
            "image_type": "czi",
            "acq_date": img.metadata.get(
                "Acquisition Date", "Unknown"
            ),  # e.g., '2025-03-19T16:32:53.6334534Z'
            "total_series": (
                img.scenes.shape[0] if hasattr(img.scenes, "shape") else 1
            ),
            "size_x": img.dims.X,
            "size_y": img.dims.Y,
            "size_z": img.dims.Z,
            "size_c": img.dims.C,
            "size_t": img.dims.T,
            "size_s": img.dims.S if hasattr(img.dims, "S") else 1,
            "size_b": img.dims.B if hasattr(img.dims, "B") else 1,
            "size_m": img.dims.M if hasattr(img.dims, "M") else 1,
            "dim_order_bf": str(img.dims.order),
            "axes_czifile": "STCYX0",  # From czifile if needed
            "shape_czifile": img.data.shape,
            #"czi_is_mosaic": img.is_mosaic,
            #"obj_na": img.physical_pixel_sizes.X,  # Or from metadata
            #"obj_mag": 10.0,  # Parse from metadata['Objective']
            #"obj_id": img.metadata.get("Objective ID", "Unknown"),
            #"obj_name": img.metadata.get("Objective Name", ["Unknown"]),
        }
        return metadata

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        img = BioImage(str(path), reader=bioio_czi.Reader)
        # Lazy slice via BioImage params
        scene_kwargs = {}
        if t is not None:
            scene_kwargs["T"] = t
        if z is not None:
            scene_kwargs["Z"] = z
        if c is not None:
            scene_kwargs["C"] = c
        data = img.get_image_dask_data(**scene_kwargs).squeeze()
        return data


@log_method
def get_loader(path: Path) -> BaseLoader:
    loaders = {
        "czi": CziLoader(),
        "imagexpress": ImageXpressLoader(),
        "sitetiff": SiteTiffLoader(),
        "phenix": PhenixLoader(),
        #"tif": TiffLoader(),
    }
    for key in loaders:
        loader = loaders[key]
        if loader.can_read(path):
            logger.debug("selected %s", str(loader))
            return loader

    raise ValueError("No loader found for " + str(path))


@log_method
def load_plate(
    directory: Path,
    file_type: str = "tif",
    format_preset: str = "auto",  # 'auto', 'imagexpress', 'czi', etc.
    wells: list[str] | None = None,
    nwells: int = -1,
    nsites: int = -1,
    iol: str = "image",  # or 'label'
    name: str = "image_or_label_name",
) -> dict[str, Any]:
    """
    Orchestrator: Load folder into plate/df/metadata.
    Returns: {'df': pd.DataFrame, 'plate': Plate, 'metadata': dict, 'aux_data': dict}
    """
    if not directory.exists():
        logger.error("Directory does not exist: %s", directory)
        return {
            "df": pd.DataFrame(),
            "plate": Plate(),
            "metadata": {},
            "aux_data": {},
        }

    # Glob all allowed files
    formats = [".czi", ".tif", ".tiff"]
    files = [p for p in directory.glob("**/*") if p.is_file() and p.suffix in formats]

    if not files:
        logger.error("No files found in directory: %s", directory)
        return {
            "df": pd.DataFrame(),
            "plate": Plate(),
            "metadata": {},
            "aux_data": {},
        }

    logger.debug("len(files) %d", len(files))

    # Pick loader (preset overrides auto)
    first_file = files[0]
    if format_preset == "auto":
        loader = get_loader(
            first_file
        )  # Registry picks ImageXpress for .tif with pattern, etc.
    else:
        # Map preset to loader (expandable)
        loader_map = {
            "imagexpress": ImageXpressLoader(),
            "phenix": PhenixLoader(),
            "sitetiff": SiteTiffLoader(),
            "czi": CziLoader(),
            "generic": TiffLoader(),
        }
        loader = loader_map.get(format_preset, TiffLoader())

    # Discover metadata/df
    df = loader.discover_metadata(files)
    if df.empty:
        return {"df": df, "plate": Plate(), "metadata": {}, "aux_data": {}}

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

    # Separate main/aux (loader-specific)
    main_df, aux_dict = loader.separate_stacks_and_aux(
        df
    )  # New method on loader (empty for generic)

    # Extra metadata (loader-specific, e.g., dims/acq_date)
    metadata = loader.get_extra_metadata(
        files[0]
    )  # New method, e.g., {'dims': img.dims, ...}

    state = StateManager.get_instance()  # Singleton access
    # reset state if new image loaded
    if iol == "image":
        state.clear_state()  # Fresh start
        state.plate = Plate()
        state.df_images = df
        state.metadata = metadata
        state.loader_img = loader
    elif iol == "label":
        state.loader_lbl = loader

    logger.debug("name %s", name)

    # Build plate (pass loader for slicing)
    plate = StateManager.get_instance().plate
    build_plate_from_df_fast(
        # {'stack': main_df, **aux_dict}, plate, iol=iol, name="image", loader=loader
        df,
        plate,
        iol=iol,
        name=name,
        # loader=loader,
    )

    return {
        "df": df,
        "plate": plate,
        "metadata": metadata,
        "aux_data": aux_dict,
    }


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


@log_method
def build_plate_from_df(
    stacks_df: pd.DataFrame,
    plate: Plate,
    iol: str = "image",
    name: str = "image_name",
    loader: BaseLoader = None,
):
    """Generic builder: Groupby, delegate stacking to loader."""
    if loader is None:
        raise ValueError("Loader required for format-specific stacking")
    grouped_df = stacks_df.groupby([WELL, SITE])

    for (well, site), site_group in tqdm(grouped_df, desc="Building sites"):
        logger.debug("well %s site %s", well, site)
        site_array = loader.build_site_array(site_group)
        logger.debug("site %s array shape: %s", site, site_array.shape)
        well_site = plate.get_well_site(well, site)
        if iol == "image":
            well_site.set_image(name, site_array)
        elif iol == "label":
            well_site.set_label_image(name, site_array)
