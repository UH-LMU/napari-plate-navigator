import logging
import platform
from pathlib import Path

import dask.array as da
import pandas as pd

from ._utils import log_method

logger = logging.getLogger(__name__)  # Module-level logger


class Site:
    def __init__(self, name: str):
        self.name = name
        self.images: dict[str, da.Array] = (
            {}
        )  # Key: image name, Value: Dask array
        self.auxiliaries: dict[str, da.Array] = (
            {}
        )  # Key: label name, Value: Dask array
        self.labels: dict[str, da.Array] = (
            {}
        )  # Key: label name, Value: Dask array

        self.filelists_img: dict[str, pd.DataFrame] = {}
        self.filelists_lbl: dict[str, pd.DataFrame] = {}

    def set_image(self, name: str, image: da.Array):
        self.images[name] = image

    def set_auxiliary(self, name: str, array: da.Array):
        self.auxiliaries[name] = array

    def set_label_image(self, name: str, label_image: da.Array):
        self.labels[name] = label_image

    def set_filelist_img(self, name: str, df: pd.DataFrame):
        self.filelists_img[name] = df

    def set_filelist_lbl(self, name: str, df: pd.DataFrame):
        self.filelists_lbl[name] = df
        logger.debug("set_filelist_lbl %s", self.filelists_lbl.keys)

    @log_method
    def get_images(self) -> dict[str, da.Array]:
        logger.debug("images.keys %s", self.images.keys())
        if len(self.images) != len(self.filelists_img):
            # build arrays if not built yet
            self.build_on_demand()
        logger.debug("images.keys %s", self.images.keys())
        return self.images

    @log_method
    def get_labels(self) -> dict[str, da.Array]:
        logger.debug("filelists_lbl.keys %s", self.filelists_lbl.keys())
        logger.debug("labels.keys %s", self.labels.keys())
        if len(self.labels) != len(self.filelists_lbl):
            # build arrays if not built yet
            self.build_on_demand()
        logger.debug("labels.keys %s", self.labels.keys())
        return self.labels

    def build_on_demand(self):
        """Lazy build on demand."""
        loader = StateManager.get_instance().loader

        for key, df in self.filelists_img.items():
            if df.empty or key in self.images:
                continue
            self.images[key] = loader.build_site_array(df)

        for key, df in self.filelists_lbl.items():
            if df.empty or key in self.labels:
                continue
            self.labels[key] = loader.build_site_array(df)

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
        self.channel_contrast: dict[str, tuple[float, float]] = (
            {}
        )  # e.g., {'Ch1': (0.0, 255.0)}
        self.saved_t: int = 0  # Selected time step
        self.saved_z: int = 0  # Selected Z slice
        self.loader = None
        self.czi = None
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

    def get_channel_contrast(self, layer_name: str) -> tuple[float, float]:
        """Get contrast limits from state (default (0, 255) if unset)."""
        return self.channel_contrast.get(layer_name, (0.0, 65535.0))

    def set_channel_contrast(
        self, layer_name: str, limits: tuple[float, float]
    ):
        """Update contrast in state."""
        self.channel_contrast[layer_name] = limits

    def get_saved_t(self) -> int:
        return self.saved_t

    def set_saved_t(self, t: int):
        self.saved_t = t  # max(0, min(t, self.plate.shape[0] - 1 if self.plate else 0))  # Clamp

    def get_saved_z(self) -> int:
        return self.saved_z

    def set_saved_z(self, z: int):
        self.saved_z = z  # max(0, min(z, self.plate.shape[1] - 1 if self.plate else 0))  # Clamp

    def clear_state(self):
        """Reset on new plate load."""
        self.plate = None
        self.df_images = pd.DataFrame()
        self.channel_vis.clear()
        self.label_vis.clear()
        self.channel_contrast.clear()
        self.loader = None


# Optional: Expose key bits for widget
__all__ = [
    "Plate",
    "StateManager",
]  # For easy imports
