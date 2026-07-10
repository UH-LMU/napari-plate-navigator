"""
Widget-level tests that drive ``NavigationWidget.update_image()`` across
multiple sites — both with and without labels loaded. These exercise the
clear/re-add cycle in the widget, which the loader-only integration tests
do not cover.

Require real data; skipped if no ``--phenix-dir`` (and ``--phenix-labels-dir``
for the labels test) is provided. See conftest.py.
"""

import pytest

from napari_plate_navigator._base import StateManager
from napari_plate_navigator._reader import SITE, WELL, load_plate
from napari_plate_navigator._widget import NavigationWidget


def _skip_if_none(path, label):
    if path is None:
        pytest.skip(f"No {label} data provided")


def _setup_widget(make_napari_viewer):
    """Create a viewer + fresh StateManager + NavigationWidget."""
    viewer = make_napari_viewer()
    state = StateManager.get_instance()
    state.clear_state()
    widget = NavigationWidget(viewer, state)
    return viewer, state, widget


def _load_images_and_populate(widget, state, image_dir):
    """Mirror what select_folder_images does in production."""
    load_plate(image_dir)
    wells = sorted(state.df_images[WELL].unique())
    widget.update_wells(wells)
    # In production, a QTimer.singleShot fires update_image() after
    # update_wells; do that explicitly here.
    widget.update_image()
    return wells


def _sites_for_well(state, well_name):
    df = state.df_images
    return sorted(str(s) for s in df[df[WELL] == well_name][SITE].unique())


def _layer_kinds(viewer):
    return [(ly.name, ly.__class__.__name__) for ly in viewer.layers]


def _has_labels(viewer):
    return any(ly.__class__.__name__ == "Labels" for ly in viewer.layers)


class TestPhenixWidgetNavigation:
    def test_navigate_sites_images_only(self, make_napari_viewer, phenix_dir):
        _skip_if_none(phenix_dir, "phenix")
        viewer, state, widget = _setup_widget(make_napari_viewer)

        wells = _load_images_and_populate(widget, state, phenix_dir)
        assert wells, "no wells discovered"

        first_well = widget.well_combo.currentText()
        sites = _sites_for_well(state, first_well)
        assert sites, "no sites in first well"

        n_image_layers = len(viewer.layers)
        assert n_image_layers > 0, "no layers after initial load"

        # Switch to a second site if available; layer count should be stable.
        if len(sites) > 1:
            widget.site_combo.setCurrentIndex(1)
            assert len(viewer.layers) == n_image_layers, (
                f"layer count changed after site switch: "
                f"{n_image_layers} -> {len(viewer.layers)}"
            )

    def test_navigate_sites_with_labels(
        self, make_napari_viewer, phenix_dir, phenix_labels_dir
    ):
        _skip_if_none(phenix_dir, "phenix images")
        _skip_if_none(phenix_labels_dir, "phenix labels")
        viewer, state, widget = _setup_widget(make_napari_viewer)

        # 1. Load images and render initial site.
        wells = _load_images_and_populate(widget, state, phenix_dir)
        assert wells

        # 2. Load labels — mirror select_folder_labels (load_plate + explicit
        #    update_image to re-render current site).
        load_plate(phenix_labels_dir, iol="label", name="cellpose")
        widget.update_image()

        # 3. Find a well that actually has labels AND multiple sites.
        target_well = None
        for w in wells:
            sites = _sites_for_well(state, w)
            if len(sites) < 2:
                continue
            widget.well_combo.setCurrentText(w)
            widget.update_image()
            if _has_labels(viewer):
                target_well = w
                break

        if target_well is None:
            pytest.skip("No well with multiple sites + labels found")

        # 4. Switch sites within that well, verifying labels and image
        #    layer counts stay stable across each switch.
        widget.well_combo.setCurrentText(target_well)
        widget.site_combo.setCurrentIndex(0)
        widget.update_image()
        layers_site0 = _layer_kinds(viewer)
        n_img_0 = sum(1 for _, k in layers_site0 if k == "Image")
        n_lbl_0 = sum(1 for _, k in layers_site0 if k == "Labels")
        assert (
            n_lbl_0 > 0
        ), f"labels missing on site 0 of {target_well}: {layers_site0}"

        widget.site_combo.setCurrentIndex(1)
        widget.update_image()
        layers_site1 = _layer_kinds(viewer)
        n_img_1 = sum(1 for _, k in layers_site1 if k == "Image")
        n_lbl_1 = sum(1 for _, k in layers_site1 if k == "Labels")
        assert n_lbl_1 > 0, (
            f"labels missing after switching to site 1 of {target_well}: "
            f"{layers_site1}"
        )
        assert n_img_0 == n_img_1, (
            f"image layer count changed across site switch: "
            f"{n_img_0} -> {n_img_1}"
        )
        assert n_lbl_0 == n_lbl_1, (
            f"label layer count changed across site switch: "
            f"{n_lbl_0} -> {n_lbl_1}"
        )

        # 5. Switch back to site 0 — labels should still be there.
        widget.site_combo.setCurrentIndex(0)
        widget.update_image()
        layers_back = _layer_kinds(viewer)
        n_lbl_back = sum(1 for _, k in layers_back if k == "Labels")
        assert n_lbl_back == n_lbl_0, (
            f"label layer count changed on return to site 0: "
            f"{n_lbl_0} -> {n_lbl_back}; layers={layers_back}"
        )
