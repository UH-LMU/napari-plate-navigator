import platform
from pathlib import Path

import dask.array as da
import pandas as pd


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
]  # For easy imports
