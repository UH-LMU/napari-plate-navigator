# src/napari_plate_navigator/_widget.py
import logging
import os
import pandas as pd
from pathlib import Path

import napari  # Import for proper Viewer annotation
from magicgui import magicgui
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

# Import core logic from _base (assuming you've ported constants/classes/functions there)
from ._base import (
    StateManager,
    get_mount_path,
)
from ._reader import SITE, WELL, load_plate
from ._utils import log_method

logger = logging.getLogger(__name__)

# Test settings
TESTING = True
if TESTING:
    user = os.getenv("USER")
    # 2D example
    default_image_path = Path(f"/home/{user}/data/pranoy/images/project1")
    default_label_path = Path(f"/home/{user}/data/pranoy/stardist/project1")
    # 3D example
    default_image_path = Path(f"/home/{user}/data/yu/images/visit20251007")
    default_label_path = Path(f"/home/{user}/data/yu/stardist/visit20251007")
    # Phenix example
    default_image_path = Path(f"/home/{user}/data/connexin")
    # CZI example
    default_image_path = Path(f"/home/{user}/data")
    default_label_path = Path(f"/home/{user}/data/andrea")
    
else:
    default_image_path = get_mount_path() / "instruments/Micro" / "project1"
    default_label_path = get_mount_path() / "airflow/micro" / "project1"


class NavigationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, state: StateManager):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.viewer = viewer
        self.state = state

        layout = QVBoxLayout()
        self.well_label = QLabel("Well")
        self.well_combo = QComboBox()
        self.well_combo.currentTextChanged.connect(self.update_sites_for_well)
        self.well_combo.currentTextChanged.connect(self.update_image)
        self.site_label = QLabel("Site")
        self.site_combo = QComboBox()
        self.site_combo.currentTextChanged.connect(self.update_image)
        layout.addWidget(self.well_label)
        layout.addWidget(self.well_combo)
        layout.addWidget(self.site_label)
        layout.addWidget(self.site_combo)
        self.setLayout(layout)
        self.wells: list[str] = []
        self.sites: list[str] = []

    def update_wells(self, wells: list[str]):
        self.wells = sorted(wells)
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        self.well_combo.addItems(self.wells)
        self.well_combo.blockSignals(False)
        if self.wells:  # Auto-select first well and update sites
            self.well_combo.setCurrentIndex(0)
            self.update_sites_for_well(self.wells[0])  # Trigger initial sites
            
    def update_sites_for_well(self, well_name: str):
        """Filter sites for selected well from state.df_images (or well_df)."""
        if well_name not in self.wells:
            return
        # General case: Filter from df_images (has WELL/SITE)
        site_df = self.state.df_images[self.state.df_images[WELL] == well_name]
        unique_sites = sorted(site_df[SITE].unique())
        self.sites = [str(s) for s in unique_sites]
        self.site_combo.clear()
        if self.sites:
            self.site_combo.blockSignals(True)
            self.site_combo.addItems(self.sites)
            # Auto-select first site if available
            self.site_combo.setCurrentIndex(0)
            self.site_combo.blockSignals(False)
            # Trigger image update
            #self.update_image()
        else:
            self.site_combo.addItem("No sites")  # Placeholder

    def update_sites(self, sites: list[int]):
        self.sites = [str(s) for s in sorted(sites)]
        self.site_combo.blockSignals(True)
        self.site_combo.clear()
        self.site_combo.addItems(self.sites)
        self.site_combo.blockSignals(False)

    @log_method
    def update_image(
        self, selected_text=None
    ):  # Accept arg from signal (optional/ignored)
        well = self.well_combo.currentText()
        site = self.site_combo.currentText()
        self.logger.info("well:%s site:%s", well, site)

        if not (well and site):
            self.logger.info("well or site missing")
        else:
            if self.state.plate:
                self.logger.debug(
                    "plate.nwells before clear: %s", self.state.plate.nwells()
                )
            self.viewer.layers.clear()
            site_obj = self.state.plate.get_well_site(well, site)
            # site_obj.debug()
            # self.state.plate.debug()

            # Add images, then restore visibility
            for _img_name, img in site_obj.get_images().items():
                dims_tuple = ('T','Z','C','Y','X')
                if hasattr(img, "dims"):
                    dims_tuple = img.dims

                # Find 'C' position (e.g., 2 in ('I', 'T', 'C', 'Z', 'Y', 'X'))
                try:
                    channel_pos = dims_tuple.index('C')
                    # Channel axis for Napari (from end: - (len - pos))
                    channel_axis = - (len(dims_tuple) - channel_pos)
                except ValueError:
                    channel_axis = -3  # Fallback for standard TZYXC
                    logger.warning("No 'C' in dims %s; using default -3",
                                   str(dims_tuple))

                # Generate generic channel names based on number of channels
                if hasattr(img, "dims"):
                    self.logger.debug("dims %s %s", _img_name, img.dims)
                self.logger.debug("shape %s %s", _img_name, img.shape)
                num_channels = img.shape[channel_axis]
                channel_names = [f"Ch{i+1}" for i in range(num_channels)]
                added_layers = self.viewer.add_image(
                    img, channel_axis=channel_axis, name=channel_names
                )
                # added_layers is always a list of Image layers when names is a list
                for added_layer in added_layers:
                    layer_name = added_layer.name
                    added_layer.visible = self.state.get_layer_visibility(
                        layer_name
                    )
                    added_layer.contrast_limits = (
                        self.state.get_channel_contrast(layer_name)
                    )  # Restore contrast
                    added_layer.events.visible.connect(
                        lambda event, name=layer_name: self.state.set_layer_visibility(
                            name, event.source.visible
                        )
                    )
                    added_layer.events.contrast_limits.connect(
                        lambda event, name=layer_name: self.state.set_channel_contrast(
                            name, event.source.contrast_limits
                        )
                    )

            # Add labels, restore visibility
            for lbl_name, lbl in site_obj.get_labels().items():
                if hasattr(lbl, "dims"):
                    self.logger.debug("dims %s %s", lbl_name, lbl.dims)
                self.logger.debug("shape %s %s", lbl_name, lbl.shape)
                added_layer = self.viewer.add_labels(lbl, name=lbl_name)
                added_layer.visible = self.state.get_layer_visibility(
                    lbl_name, is_label=True
                )
                # Hook event for future changes (use event.source.visible for the new value)
                added_layer.events.visible.connect(
                    lambda event, name=lbl_name: self.state.set_layer_visibility(
                        name, event.source.visible, is_label=True
                    )
                )

            # Restore T/Z selection (after dims labels set)
            # These don't work, investigate later if really needed.
            # self.viewer.dims.current_step[0] = self.state.get_saved_t()  # T
            # self.viewer.dims.current_step[1] = self.state.get_saved_z()  # Z

            # For now, set all indices to 0.
            for i in range(len(self.viewer.dims.point)):
                self.viewer.dims.set_point(i, 0)

            self.logger.info(
                "plate.nwells after: %d", self.state.plate.nwells()
            )


class FolderSelectors(QWidget):
    def __init__(self, select_folder_images, select_folder_labels):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(select_folder_images.native)
        layout.addWidget(select_folder_labels.native)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setLayout(layout)


def make_qwidget() -> NavigationWidget:
    # Get the current viewer (npe2 doesn't always inject reliably in some setups)
    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("No Napari viewer found. Ensure Napari is running.")

    state = StateManager.get_instance()  # Singleton access
    state.clear_state()  # Fresh start

    navigate_widget = NavigationWidget(viewer, state)

    # Hook dims changes to save T/Z state
    # TODO: there is something wrong with this.
    # For now, set all indices to 0. Leave this here in case needed later.
    def on_dims_changed(event):
        current = viewer.dims.current_step
        state.set_saved_t(current[0])  # T axis 0
        state.set_saved_z(current[1])  # Z axis 1

    viewer.dims.events.current_step.connect(on_dims_changed)

    @magicgui(
        folder={
            "label": "Select Folder (images)",
            "mode": "d",
            "value": default_image_path,
        },
        auto_call=True,
    )
    def select_folder_images(folder: Path):
        print("select_folder_images: clear_state")

        _result = load_plate(
            folder, file_type="tif"
        )  # Or 'tif' for ImageXpress

        print("*****")
        print("*****")
        print("***** New image loaded. ")
        print(state.df_images.head())
        print("*****")
        print("*****")

        try:
            wells_list = sorted(state.df_images[WELL].unique())
            navigate_widget.update_wells(wells_list)
        except KeyError:
            # stop here if state df doesn't have WELL column
            return

        # Manually trigger update_image after setting combos to load initial view
        # This ensures the first well/site combo is processed after population
        QTimer.singleShot(0, navigate_widget.update_image)

        viewer.dims.axis_labels = [
            "TimeStep",
            "Z-slice",
            "Channel",
            "Y",
            "X",
        ]  # Adjust based on shape
        for i in range(len(viewer.dims.point)):
            viewer.dims.set_point(i, 0)

    @magicgui(
        label_name={"label": "Label Name (e.g., nuclei)"},
        file_type={"label": "File Type", "value": "tif"},
        folder={
            "label": "Select Folder (labels)",
            "mode": "d",
            "value": default_label_path,
        },
        call_button="Load Labels",
    )
    @log_method
    def select_folder_labels(label_name: str, file_type: str, folder: Path):
        # state = StateManager.get_instance()  # Singleton access
        logger.debug("label_name %s path %s", label_name, str(folder))
        if not folder or not folder.exists():
            logger.error("Invalid folder")
            return

        _result = load_plate(
            folder, file_type="tif", iol="label", name=label_name
        )  # Or 'tif' for ImageXpress
        # Handle missing labels if needed (using the nonlocal df_images)
        # df_labels = handle_missing_labels(df_images, df_labels, file_type)  # Uncomment when implemented

        # Refresh the current view after loading new labels
        QTimer.singleShot(0, navigate_widget.update_image)

    # Setup docks after defining magicguis
    folder_selectors = FolderSelectors(
        select_folder_images, select_folder_labels
    )
    viewer.window.add_dock_widget(folder_selectors, area="right")
    viewer.window.add_dock_widget(navigate_widget, area="right")

    viewer.window._qt_window.setWindowTitle(
        f"Napari - {viewer.window._qt_window.windowTitle()} - Plate Navigator"
    )

    return navigate_widget


# Optional: Auto-trigger initial load if defaults exist, but keep manual for now
