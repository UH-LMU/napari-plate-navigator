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

Always use `pixi run --environment testing` — plain `pytest` resolves to the system
Python which has none of the project's packages.

```bash
# Unit tests (no data needed, always run, ~0.2s)
pixi run --environment testing test-unit

# Per-format integration tests
pixi run --environment testing test-ix      --imagexpress-dir ./data/imagexpress
pixi run --environment testing test-phenix  --phenix-dir /DISKS/2TB/.../02-zproj/184_EXP7_P2
pixi run --environment testing test-czi     --czi-dir /DISKS/1TB/hajaalin/data/parijat/1/
pixi run --environment testing test-sitetiff  # set TEST_DATA_SITETIFF env var

# Or via environment variables (useful for CI)
TEST_DATA_PHENIX=/path/to/phenix pixi run --environment testing test-phenix

# Everything
pixi run --environment testing test
```

Note: CZI test takes ~2 minutes (CZI metadata parsing is inherently slow).

## Git workflow

- Pre-commit hooks are active (ruff, black, end-of-file-fixer, napari-plugin-checks). Run
  `pixi run --environment testing lint` before committing. Do NOT use `--no-verify`.
- Single active branch: `main`. Use short-lived feature branches for new work, merge to `main` when ready.
- Remote branches `origin/czi1`, `origin/phenix`, `origin/main` not yet updated; push `main` and
  delete `origin/phenix` and `origin/czi1` once SSH keys are available.

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
| `SiteTiffLoader` | One TIFF per site | regex: `well_.*_site_\d\d\d.*tif` |
| `SiteTiffLabelLoader` | One TIFF per site (labels) | same regex — never reached by auto-detect |
| `PhenixLoader` | Opera Phenix | regex: `r\d\dc\d\df\d\dp\d\d-ch\d` in filename |

`load_plate(directory, iol="image"|"label", name=...)` is the main entry point.
It globs recursively for `.czi`, `.png`, `.tif`, `.tiff`. When `iol="label"` and the
detected loader is `SiteTiffLoader`, it is automatically swapped to `SiteTiffLabelLoader`.

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
- `CziLoader.build_site_array` returns `xarray.DataArray` (from `get_xarray_dask_stack()`),
  not a bare `da.Array` — the widget handles this via `hasattr(img, "dims")` check
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

- [ ] Test Phenix label loading (cellpose masks may now be ready in `07-cellpose/`)
- [ ] Test CZI file 2: `/DISKS/1TB/hajaalin/data/parijat/2/`
- [ ] Prepare SiteTiff test data and run `test-sitetiff`
- [ ] Clean up repo root: `.gitignore` conda env files, scratch notebooks, backup files
- [ ] Push branches and delete remote `origin/czi1` once SSH keys are available
- [ ] Decide whether `SiteTiffLoader` label display bug ("labels still show messed up") is fixed
- [ ] Host test data on Allas CSC for CI (manifest-based HTTP download via `TEST_DATA_URL`)
- [ ] Merge `phenix` branch into `main` once all remaining items above are resolved
