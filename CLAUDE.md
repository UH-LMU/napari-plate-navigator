# napari-plate-navigator — Developer Guide for Claude

## What this is

A Napari plugin for browsing well-plate microscopy data from lab instruments.
Core use cases:
- Navigate wells and sites via dropdowns (no manual file management)
- View channel combinations without creating new composite images
- Overlay segmentation labels on original images

Supported instruments: Molecular Devices ImageXpress Micro, PerkinElmer Opera Phenix,
Zeiss CellDiscoverer 7 (CZI format). Generic TIFF-per-site also supported.

## Running tests

```bash
# Unit tests (no data needed, always run)
pixi run --environment testing pytest src/napari_plate_navigator/_tests/test_loaders_unit.py -v

# Integration tests — provide data via per-format options or env vars
pixi run --environment testing pytest src/napari_plate_navigator/_tests/test_integration.py -v \
  --imagexpress-dir ./data/imagexpress \
  --phenix-dir /path/to/phenix/measurement \
  --phenix-labels-dir /path/to/cellpose/labels \
  --czi-dir /path/to/czi/files

# Or via environment variables
TEST_DATA_IMAGEXPRESS=./data/imagexpress \
TEST_DATA_PHENIX=/path/to/phenix \
pixi run --environment testing pytest ...
```

## Git workflow

- Always use `--no-verify` on `git commit` (pre-commit hooks are from the napari template
  and not enforced here)
- Active branches: `main` (canonical) and `phenix` (kept in sync with main)
- Remote branches `origin/czi1`, `origin/phenix`, `origin/main` exist; push when keys available

## Architecture

### Data model (`_base.py`)
```
Plate → Well → Site
                ├── images: dict[name → dask.Array]      (lazy)
                ├── labels: dict[name → dask.Array]       (lazy)
                ├── filelists_img: dict[name → DataFrame]  (metadata, built eagerly)
                └── filelists_lbl: dict[name → DataFrame]
```
Arrays are built on demand in `Site.build_on_demand()` via the loader stored in `StateManager`.

### Loader system (`_reader.py`)

`get_loader(path)` tries loaders in order until `can_read()` returns True:

| Loader | Format | Detection |
|--------|--------|-----------|
| `CziLoader` | Zeiss CZI | `.czi` suffix |
| `ImageXpressLoader` | Molecular Devices IXM | regex: `t\d+_\w\d{2}_s\d+_w\d_z\d+` in path |
| `SiteTiffLabelLoader` | One TIFF per site (labels) | regex: `well_.*_site_\d\d\d.*tif` |
| `SiteTiffLoader` | One TIFF per site (images) | regex: `well_.*_site_\d\d\d.*tif` |
| `PhenixLoader` | Opera Phenix | regex: `r\d\dc\d\df\d\dp\d\d-ch\d` in filename |

`load_plate(directory, iol="image"|"label", name=...)` is the main entry point.
It globs recursively for `.czi`, `.png`, `.tif`, `.tiff`.

Key `StateManager` fields: `plate`, `df_images`, `loader_img`, `loader_lbl`, `czi` (BioImage for CZI).

### Widget (`_widget.py`)

`make_qwidget()` creates the plugin UI:
- `NavigationWidget`: Well + Site dropdowns, calls `update_image()` on change
- `select_folder_images`: magicgui auto-call widget, triggers `load_plate()`
- `select_folder_labels`: magicgui manual-call widget

### Known quirks

- `ImageXpressLoader.build_site_array`: after `DataFrame.explode()`, int columns
  become object dtype — always apply `pd.to_numeric()` afterwards
- Z-steps in ImageXpress data are 1-indexed; use `nunique()` not `max()+1` for `full_z`
- `SiteTiffLabelLoader.build_site_array` drops C dimension (`.isel(C=0)`) so labels
  have the right shape for Napari
- Default paths in `_widget.py` are hardcoded for Harri's machines (TESTING=True block)
- `StateManager` is a singleton — call `state.clear_state()` before loading new images

## Test data locations (local, not in git)

| Format | Path |
|--------|------|
| ImageXpress | `./data/imagexpress/` |
| Phenix images | `/DISKS/2TB/hajaalin/data/connexin/nextflow_pub/02-zproj/184_EXP7_P2` |
| Phenix labels (cellpose) | `/DISKS/2TB/hajaalin/data/connexin/nextflow_pub/07-cellpose/labels/184_EXP7_P2` |
| CZI (file 1) | `/DISKS/1TB/hajaalin/data/parijat/1/notFAILS20240814_00002-04.czi` |
| CZI (file 2) | `/DISKS/1TB/hajaalin/data/parijat/2/e20240731_AwesomeFails-03.czi` |

Future: test data will be hosted on Allas CSC object storage (S3-compatible public URLs).
The conftest supports `TEST_DATA_URL` for HTTP download via a manifest file.

## Remaining work

- [ ] Verify CZI loader works end-to-end with parijat data
- [ ] Test Phenix label loading once cellpose masks finish processing
- [ ] Clean up repo root: `.gitignore` conda env files, scratch notebooks, backup files
- [ ] Push branches once SSH keys are available on this machine
- [ ] Decide whether `SiteTiffLoader` label display bug ("labels still show messed up") is fixed
