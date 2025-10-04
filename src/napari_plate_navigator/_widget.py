# src/napari_plate_navigator/_widget.py
import time
from pathlib import Path

import napari  # Import for proper Viewer annotation
import pandas as pd
from magicgui import magicgui
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

# Import core logic from _base (assuming you've ported constants/classes/functions there)
from ._base import (
    SITE,
    WELL,
    Plate,
    create_file_list,
    get_mount_path,
    get_stacks_and_projections,
    load_dask_array,
)

# Test settings
TESTING = True
if TESTING:
    user = "user"  # Or import os; os.getenv('USER')
    default_image_path = Path(f"/home/{user}/data/pranoy/images/project1")
    default_label_path = Path(f"/home/{user}/data/pranoy/stardist/project1")
else:
    default_image_path = get_mount_path() / "instruments/Micro" / "project1"
    default_label_path = get_mount_path() / "airflow/micro" / "project1"


class NavigationWidget(QWidget):
    def __init__(
        self, viewer: napari.Viewer, plate: Plate, df_images: pd.DataFrame
    ):
        super().__init__()
        self.viewer = viewer
        self.plate = plate  # Store reference to plate
        self.df_images = df_images  # Store if needed for labels

        layout = QVBoxLayout()
        self.well_label = QLabel("Well")
        self.well_combo = QComboBox()
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
        self.well_combo.clear()
        self.well_combo.addItems(self.wells)

    def update_sites(self, sites: list[int]):
        self.sites = [str(s) for s in sorted(sites)]
        self.site_combo.clear()
        self.site_combo.addItems(self.sites)

    def update_image(self):
        well = self.well_combo.currentText()
        site = self.site_combo.currentText()
        print(f"well {well} site {site}")
        if well and site:
            print("jee")
            ###
            ## Now plate should be populated via closure
            ###
            print(f"plate.nwells before clear: {self.plate.nwells()}")
            self.viewer.layers.clear()
            site_obj = self.plate.get_well_site(well, site)
            site_obj.debug()
            self.plate.debug()
            for _img_name, img in site_obj.images.items():
                # Generate generic channel names based on number of channels (assuming channel dim at index -3)
                num_channels = img.shape[-3]
                channel_names = [f"Ch{i+1}" for i in range(num_channels)]
                self.viewer.add_image(img, channel_axis=-3, name=channel_names)
            for lbl_name, lbl in site_obj.labels.items():
                self.viewer.add_labels(lbl, name=lbl_name)
            print(f"plate.nwells after: {self.plate.nwells()}")


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

    """Factory function: Returns the main widget and sets up docks."""
    # Your original main logic, minus click and napari.run()
    well_list = []  # Default to ALL; could add a param later
    nwells, nsites = -1, -1  # Defaults

    print("Plugin: init plate")
    plate = Plate()  # From _base
    df_images = pd.DataFrame()  # Initialize

    navigate_widget = NavigationWidget(viewer, plate, df_images)

    @magicgui(
        folder={
            "label": "Select Folder (images)",
            "mode": "d",
            "value": default_image_path,
        },
        auto_call=True,
    )
    def select_folder_images(folder: Path):
        nonlocal plate, df_images
        print("select_folder_images: reset plate")
        plate = Plate()  # Reset plate
        navigate_widget.plate = plate  # Update the widget's reference

        df_images = create_file_list(
            folder, wells=well_list, nwells=nwells, nsites=nsites
        )
        if df_images.empty:
            print("No images found")
            return

        stacks, projs = get_stacks_and_projections(df_images)

        if not stacks.empty:
            t_start = time.time()
            grouped = stacks.groupby(by=[WELL, SITE]).agg(list)
            load_dask_array(grouped, plate, iol="image", name="image")
            print(f"Loaded images in {time.time() - t_start:.2f}s")
            print(f"plate.nwells after load: {plate.nwells()}")

            # TODO: Handle projections if needed

            wells_list = list(stacks[WELL].unique())
            sites_list = list(stacks[SITE].unique())
            navigate_widget.update_wells(wells_list)
            navigate_widget.update_sites(sites_list)

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
    def select_folder_labels(label_name: str, file_type: str, folder: Path):
        nonlocal plate, df_images
        if not folder or not folder.exists():
            print("Invalid folder")
            return

        df_labels = create_file_list(
            folder,
            file_type=file_type,
            wells=well_list,
            nwells=nwells,
            nsites=nsites,
        )
        if df_labels.empty:
            print("No labels found")
            return

        # Handle missing labels if needed (using the nonlocal df_images)
        # df_labels = handle_missing_labels(df_images, df_labels, file_type)  # Uncomment when implemented

        grouped = df_labels.groupby(by=[WELL, SITE]).agg(list)
        t_start = time.time()
        load_dask_array(grouped, plate, iol="label", name=label_name)
        print(f"Loaded labels '{label_name}' in {time.time() - t_start:.2f}s")

        # Refresh the current view after loading new labels
        QTimer.singleShot(0, navigate_widget.update_image)

        for i in range(len(viewer.dims.point)):
            viewer.dims.set_point(i, 0)

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
