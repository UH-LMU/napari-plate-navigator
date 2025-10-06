# src/napari_plate_navigator/_reader.py
import logging
from pathlib import Path
from typing import Any

import dask.array as da
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers import CziReader  # For explicit CZI support
from tqdm import tqdm

from ._base import Plate
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

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Extract T/Z/C/WELL/SITE/PLATE from files/metadata, return df."""
        raise NotImplementedError

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        """Lazy-load a slice as dask array."""
        raise NotImplementedError

    def separate_stacks_and_aux(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        return df, {}  # Generic: No aux

    def get_extra_metadata(self, path: Path) -> dict[str, Any]:
        return {}  # Generic


class TiffLoader0(BaseLoader):
    #    """Fallback for TIFF/PNG multi-file (your current MolDev logic)."""
    #    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
    #        return _base.create_file_list(Path(files[0].parent), 'tif')  # Reuse existing

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        img = AICSImage(str(path))
        #        data = img.get_image_dask_data().squeeze()
        #        if t is not None:
        #            data = data[t]
        #        if z is not None:
        #            data = data[:, z]  # Assume Z after T
        #        if c is not None:
        #            data = data[:, :, c]
        data = img.get_image_dask_data()
        return data


class TiffLoader(BaseLoader):
    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Simple df from paths; no parsing—fallback for non-structured files."""
        df = pd.DataFrame({"PATH": [str(f) for f in files]})
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
        img = AICSImage(str(path))
        # Lazy slice via AICSImage params (works for embedded dims or single)
        scene_kwargs = {}
        if t is not None:
            scene_kwargs["T"] = t
        if z is not None:
            scene_kwargs["Z"] = z
        if c is not None:
            scene_kwargs["C"] = c
        data = img.get_image_dask_data(**scene_kwargs).squeeze()
        return data


class ImageXpressLoader(TiffLoader):
    """Specific loader for Molecular Devices ImageXpress (regex + proj logic)."""

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
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
            r"[/\\]t(?P<{mc2}>\d+)_(?P<{mc5}>\w\d{{2}})_s(?P<{mc6}>\d{{1,2}})_(?P<{mc7}>w\d)_z(?P<{mc3}>\d+)"
        ).format(**metadata_columns)
        extracted = df[PATH].str.extract(pattern)
        df = df.join(extracted)

        df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
        df[PLATE] = df[PLATE].astype(str)
        df[WELL] = df[WELL].astype(str)
        df[SITE] = df[SITE].astype(int)
        df[CHANNEL] = df[CHANNEL].astype(str)
        df[TSTEP] = df[TSTEP].astype(int)
        df[ZSTEP] = df[ZSTEP].astype(int)

        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        return df

    def separate_stacks_and_aux(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        # Your get_stacks_and_projections code here
        ch_z = df[df[ZSTEP] > 1][CHANNEL].unique()
        ch_p = df[df[ZSTEP] == 0][CHANNEL].unique()
        assert set(ch_z) == set(
            ch_p
        ), "Mismatch between channels with slices and projections"
        projs = (
            df[df[PATH].str.contains("_Projection/")]
            .copy()
            .reset_index(drop=True)
        )
        projs.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        if len(ch_z) > 0:
            stacks = df[df[CHANNEL].isin(ch_z)].copy().reset_index(drop=True)
        else:
            stacks = df[df[ZSTEP] == 1].copy().reset_index(drop=True)
            stacks.sort_values(
                by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
                inplace=True,
                ignore_index=True,
            )
        return stacks, {"projection": projs}  # Aux dict


class CziLoader(BaseLoader):
    """Loader for Zeiss CZI files (monolithic or multi-scene)."""

    def discover_metadata(self, files: list[Path]) -> pd.DataFrame:
        """Parse filename for WELL/SITE/PLATE, embedded for T/Z/C."""
        if not files:
            return pd.DataFrame()
        path = files[0]  # Assume single file for now; extend for multi later
        img = AICSImage(str(path), reader=CziReader)

        # Filename parsing (e.g., "Plate1_WellA1_Site1_TimeSeries.czi")
        name = path.stem
        import re

        plate_match = re.search(r"Plate(?P<plate>\d+)", name)
        well_match = re.search(r"Well(?P<well>[A-P]\d{1,2})", name)
        site_match = re.search(r"Site(?P<site>\d+)", name)

        # Embedded dims (AICSImage exposes T/Z/C counts)
        t_max = img.dims.T if img.dims.T > 1 else 1
        z_max = img.dims.Z if img.dims.Z > 1 else 1
        c_max = img.dims.C

        # Build df with cartesian product for T/Z/C
        rows = []
        for t in range(t_max):
            for z in range(z_max):
                for c in range(c_max):
                    row = {
                        PATH: str(path),
                        PLATE: (
                            plate_match.group("plate") if plate_match else "1"
                        ),
                        WELL: well_match.group("well") if well_match else "A1",
                        SITE: (
                            int(site_match.group("site")) if site_match else 1
                        ),
                        TSTEP: t,
                        ZSTEP: z,
                        CHANNEL: c,
                        DIR: str(path.parent),
                    }
                    rows.append(row)
        df = pd.DataFrame(rows)
        df = df.astype(
            {
                PLATE: str,
                WELL: str,
                SITE: int,
                TSTEP: int,
                ZSTEP: int,
                CHANNEL: str,
            }
        )
        df.sort_values(
            by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
            inplace=True,
            ignore_index=True,
        )
        return df

    def get_extra_metadata(self, path: Path) -> dict[str, Any]:
        img = AICSImage(str(path), reader=CziReader)
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
            "sizes_bf": img.dims.to_tuple(),  # Full BF order
            "dim_order_bf": str(img.dims.order),
            "axes_czifile": "STCYX0",  # From czifile if needed
            "shape_czifile": img.data.shape,
            "czi_is_rgb": img.is_RGB,
            "czi_is_mosaic": img.is_mosaic,
            "obj_na": img.physical_pixel_sizes.X,  # Or from metadata
            "obj_mag": 10.0,  # Parse from metadata['Objective']
            "obj_id": img.metadata.get("Objective ID", "Unknown"),
            "obj_name": img.metadata.get("Objective Name", ["Unknown"]),
        }
        return metadata

    def load_slice(
        self, path: Path, t: int = None, z: int = None, c: int = None
    ) -> da.Array:
        img = AICSImage(str(path), reader=CziReader)
        # Lazy slice via AICSImage params
        scene_kwargs = {}
        if t is not None:
            scene_kwargs["T"] = t
        if z is not None:
            scene_kwargs["Z"] = z
        if c is not None:
            scene_kwargs["C"] = c
        data = img.get_image_dask_data(**scene_kwargs).squeeze()
        return data


# Simple registry (expandable)
LOADERS = {
    ".czi": CziLoader(),
    ".tif": ImageXpressLoader(),
    # Add .png for Phenix, etc.
}


def get_loader(path: Path) -> BaseLoader:
    ext = path.suffix.lower()
    return LOADERS.get(ext, TiffLoader())  # Fallback to TIFF logic


@log_method
def load_plate(
    directory: Path,
    file_type: str = "tif",
    format_preset: str = "auto",  # 'auto', 'imagexpress', 'czi', etc.
    wells: list[str] | None = None,
    nwells: int = -1,
    nsites: int = -1,
    iol: str = "image",  # or 'label'
) -> dict[str, Any]:
    """
    Orchestrator: Load folder into plate/df/metadata.
    Returns: {'df': pd.DataFrame, 'plate': Plate, 'metadata': dict, 'aux_data': dict}
    """
    if not directory.exists():
        return {
            "df": pd.DataFrame(),
            "plate": Plate(),
            "metadata": {},
            "aux_data": {},
        }

    # Glob files based on type
    if file_type in ["tif", "tiff", "png"]:
        files = [
            p
            for p in directory.glob(f"**/*.{file_type}")
            if "thumb" not in p.name
        ]
    else:
        files = (
            [directory] if directory.suffix.lower() == f".{file_type}" else []
        )

    if not files:
        return {
            "df": pd.DataFrame(),
            "plate": Plate(),
            "metadata": {},
            "aux_data": {},
        }

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
            CHANNEL: str,
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
    stacks, aux_data = loader.separate_stacks_and_aux(
        df
    )  # New method on loader (empty for generic)

    # Build plate (pass loader for slicing)
    plate = Plate()
    build_plate_from_df(
        stacks, plate, iol=iol, name="image", loader=loader
    )  # Updated builder takes loader

    # Extra metadata (loader-specific, e.g., dims/acq_date)
    metadata = loader.get_extra_metadata(
        files[0]
    )  # New method, e.g., {'dims': img.dims, ...}

    return {
        "df": df,
        "plate": plate,
        "metadata": metadata,
        "aux_data": aux_data,
    }


@log_method
def build_plate_from_df(
    stacks_df: pd.DataFrame,
    plate: Plate,
    iol: str = "image",
    name: str = "image_name",
    loader: BaseLoader = None,
):
    """Build plate hierarchy from df, loader-aware for multi/single-file."""
    grouped_df = stacks_df.groupby(by=[WELL, SITE]).agg(list)

    for well, well_group in grouped_df.groupby(WELL):
        for site, site_group in tqdm(
            well_group.groupby(SITE), desc=f"Building well {well}"
        ):
            exploded = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])

            t_steps = []
            for tstep, t_group in exploded.groupby(TSTEP):
                z_steps = []
                for zstep, z_group in t_group.groupby(ZSTEP):
                    channels = []

                    # for ch_path in z_group[PATH]:
                    #   c = int(z_group[CHANNEL].iloc[0])  # Convert 'w1' to 1 (HCS str to int)
                    #    if loader:
                    #        data = loader.load_slice(ch_path, t=tstep, z=zstep, c=c)
                    #    else:
                    #        # Fallback (original)
                    #        img = AICSImage(ch_path)
                    #        data = img.get_image_dask_data().squeeze()
                    #    channels.append(data)

                    # unique_paths = z_group[PATH].unique()  # Dedupe for single-file
                    for (
                        ch_idx,
                        row,
                    ) in z_group.iterrows():  # Or loop unique if multi
                        path = Path(row[PATH])
                        if (
                            loader
                            and hasattr(loader, "is_single_file")
                            and loader.is_single_file
                        ):
                            # Single-file (e.g., CZI): Slice embedded dims from one path
                            logger.info("using single-file loader")
                            data = loader.load_slice(
                                path, t=tstep, z=zstep, c=row[CHANNEL]
                            )
                        else:
                            # Multi-file fallback (ImageXpress): Load per path
                            if loader:
                                logger.info("using multi-file loader")
                                c = ch_idx[1]
                                logger.debug(
                                    "t:%d z:%d ch_idx:%s c:%d",
                                    tstep,
                                    zstep,
                                    ch_idx,
                                    c,
                                )
                                data = loader.load_slice(
                                    path, t=tstep, z=zstep, c=c
                                )
                            else:
                                logger.info("using AICSImage directly")
                                img = AICSImage(str(path))
                                data = img.get_image_dask_data().squeeze()
                        channels.append(data)
                    z_stack = da.stack(channels, axis=0)
                    z_steps.append(z_stack)
                t_stack = da.stack(z_steps, axis=0)
                t_steps.append(t_stack)

            final_array = da.stack(t_steps, axis=0)
            logger.debug("final_array.shape %s", final_array.shape)

            well_site = plate.get_well_site(well, site)
            if iol == "image":
                well_site.set_image(name, final_array)
            elif iol == "label":
                well_site.set_label_image(name, final_array)
            else:
                raise ValueError("iol must be 'image' or 'label'")
    # plate.debug()
