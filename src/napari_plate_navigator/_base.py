import platform
from pathlib import Path

import dask.array as da
import pandas as pd
from aicsimageio import AICSImage
from tqdm import tqdm

# Constants
DIR = "Directory"
PATH = "Path"
PLATE = "Plate"
WELL = "Well"
SITE = "Site"
CHANNEL = "Channel"
TSTEP = "TStep"
ZSTEP = "ZStep"


class Site:
    def __init__(self, name: str):
        self.name = name
        self.images: dict[str, da.Array] = (
            {}
        )  # Key: image name, Value: Dask array
        self.labels: dict[str, da.Array] = (
            {}
        )  # Key: label name, Value: Dask array

    def set_image(self, name: str, image: da.Array):
        self.images[name] = image

    def set_label_image(self, name: str, label_image: da.Array):
        self.labels[name] = label_image

    def debug(self):
        print(f"SITE {self.name}")
        print(f"images: {list(self.images.keys())}")
        print(f"ishapes: {[i.shape for i in self.images.values()]}")
        print(f"labels: {list(self.labels.keys())}")
        print(f"lshapes: {[i.shape for i in self.labels.values()]}")


class Well:
    def __init__(self, name: str):
        self.name = name
        self.sites: dict[str, Site] = {}

    def get_site(self, name: str) -> Site:
        name = str(name)  # Force name to str
        if name not in self.sites:
            self.sites[name] = Site(name)
        return self.sites[name]

    def get_site_names(self) -> list[str]:
        return sorted(self.sites.keys())

    def debug(self):
        print(f"WELL {self.name}")
        for site_name in sorted(self.sites.keys()):
            self.sites[site_name].debug()


class Plate:
    def __init__(self):
        self.wells: dict[str, Well] = {}

    def get_well(self, well_name: str) -> Well:
        if well_name not in self.wells:
            self.wells[well_name] = Well(well_name)
        return self.wells[well_name]

    def get_well_site(self, well_name: str, site_name: str) -> Site:
        well = self.get_well(well_name)
        return well.get_site(site_name)

    def nwells(self) -> int:
        return len(self.wells)

    def debug(self):
        print("PLATE.debug start")
        for well_name in sorted(self.wells.keys()):
            self.wells[well_name].debug()
        print("PLATE.debug end")
        print()


def get_mount_path() -> Path:
    current_os = platform.system()
    base_path = "lmu_active1"
    if current_os == "Windows":
        return Path(f"L:\\{base_path}")
    elif current_os == "Linux":
        return Path(f"/mnt/{base_path}")
    elif current_os == "Darwin":
        return Path(f"/Volumes/{base_path}")
    else:
        raise ValueError(f"Unsupported operating system: {current_os}")


def create_file_list(
    directory: Path,
    file_type: str = "tif",
    wells: list[str] | None = None,
    nwells: int = -1,
    nsites: int = -1,
) -> pd.DataFrame:
    if not directory or not directory.exists():
        return pd.DataFrame()

    files = [
        str(p)
        for p in directory.glob(f"**/*.{file_type}")
        if "thumb" not in p.name
    ]
    if not files:
        return pd.DataFrame()

    df = pd.DataFrame(files, columns=[PATH])

    # Metadata extraction regex (configurable for different instruments)
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

    if wells:
        df = df[df[WELL].isin(wells)]
    elif nwells > 0:
        unique_wells = df[WELL].unique()[:nwells]
        df = df[df[WELL].isin(unique_wells)]

    if nsites > 0:
        df = df[df[SITE] <= nsites]

    return df


def get_stacks_and_projections(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Channels with z-slices (ZStep > 1)
    ch_z = df[df[ZSTEP] > 1][CHANNEL].unique()

    # Channels with projections (ZStep == 0)
    ch_p = df[df[ZSTEP] == 0][CHANNEL].unique()

    assert set(ch_z) == set(
        ch_p
    ), "Mismatch between channels with slices and projections"

    # Separate projections (paths containing '_Projection/')
    projs = (
        df[df[PATH].str.contains("_Projection/")].copy().reset_index(drop=True)
    )
    projs.sort_values(
        by=[PLATE, WELL, SITE, TSTEP, CHANNEL], inplace=True, ignore_index=True
    )

    # Stacks: either channels with z-slices or ZStep == 1 if no z-channels
    if len(ch_z) > 0:
        stacks = df[df[CHANNEL].isin(ch_z)].copy().reset_index(drop=True)
    else:
        stacks = df[df[ZSTEP] == 1].copy().reset_index(drop=True)
    stacks.sort_values(
        by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL],
        inplace=True,
        ignore_index=True,
    )

    return stacks, projs


def build_plate_from_df(
    stacks_df: pd.DataFrame,
    plate: Plate,
    iol: str = "image",
    name: str = "image_name",
):
    grouped_df = stacks_df.groupby(by=[WELL, SITE]).agg(list)

    for well, well_group in grouped_df.groupby(WELL):
        for site, site_group in tqdm(
            well_group.groupby(SITE), desc=f"Loading well {well}"
        ):
            exploded = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])

            t_steps = []
            for _tstep, t_group in exploded.groupby(TSTEP):
                z_steps = []
                for _zstep, z_group in t_group.groupby(ZSTEP):
                    channels = []
                    for path in z_group[PATH]:
                        img = AICSImage(path)
                        data = img.get_image_dask_data().squeeze()
                        channels.append(data)
                    z_stack = da.stack(channels, axis=0)
                    z_steps.append(z_stack)
                t_stack = da.stack(z_steps, axis=0)
                t_steps.append(t_stack)

            final_array = da.stack(t_steps, axis=0)
            print(f"final_array.shape {final_array.shape}")

            well_site = plate.get_well_site(well, site)
            if iol == "image":
                well_site.set_image(name, final_array)
            elif iol == "label":
                well_site.set_label_image(name, final_array)
            else:
                raise ValueError("iol must be 'image' or 'label'")
    plate.debug()
    print("load_dask_array return")


# TODO: Implement handling for missing labels if needed
def handle_missing_labels(
    df_images: pd.DataFrame, df_labels: pd.DataFrame, file_type: str
) -> pd.DataFrame:
    # Placeholder for now; implement saving dummy labels if required
    return df_labels


class StateManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.plate: Plate | None = None
        self.df_images = pd.DataFrame()
        self.channel_vis: dict[str, bool] = (
            {}
        )  # e.g., {'Ch1': True, 'Ch2': False}
        self.label_vis: dict[str, bool] = {}  # e.g., {'nuclei': True}
        self._initialized = True

    def get_instance() -> "StateManager":
        return StateManager()

    def set_layer_visibility(
        self, layer_name: str, visible: bool, is_label: bool = False
    ):
        """Update visibility in state (call from layer events)."""
        vis_dict = self.label_vis if is_label else self.channel_vis
        vis_dict[layer_name] = visible

    def get_layer_visibility(
        self, layer_name: str, is_label: bool = False
    ) -> bool:
        """Get visibility from state (default True if unset)."""
        vis_dict = self.label_vis if is_label else self.channel_vis
        return vis_dict.get(layer_name, True)

    def clear_state(self):
        """Reset on new plate load."""
        self.plate = None
        self.df_images = pd.DataFrame()
        self.channel_vis.clear()
        self.label_vis.clear()


# Optional: Expose key bits for widget
__all__ = [
    "Plate",
    "StateManager",
    "create_file_list",
    "build_plate_from_df",
]  # For easy imports
