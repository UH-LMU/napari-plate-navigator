"""
Test fixtures for napari-plate-navigator.

Test data is never stored in git. Provide it via one of:

  Unified root (expects subdirs imagexpress/, phenix/, czi/, sitetiff/, sitetiff_labels/):
    --data-dir <path>          pytest CLI option
    TEST_DATA_DIR=<path>       environment variable
    TEST_DATA_URL=<url>        HTTP URL; files listed in a manifest are downloaded

  Per-format overrides (take precedence over the unified root):
    --imagexpress-dir / TEST_DATA_IMAGEXPRESS
    --phenix-dir      / TEST_DATA_PHENIX
    --phenix-labels-dir / TEST_DATA_PHENIX_LABELS
    --czi-dir         / TEST_DATA_CZI
    --sitetiff-dir    / TEST_DATA_SITETIFF
    --sitetiff-labels-dir / TEST_DATA_SITETIFF_LABELS

Expected directory layout for imagexpress:
  <dir>/
      <plate>/
          t001_A01_s01_w1_z001.tif
          ...

Expected directory layout for phenix images and labels:
  <dir>/
      <well>/               (e.g. r04c07/)
          <subdir>/         (e.g. projs/ or masks/)
              r04c07f01p00-ch1sk1fk1fl1_zmean.tiff
              ...
"""

import os
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Path to local test data root directory.",
    )
    parser.addoption("--imagexpress-dir", action="store", default=None)
    parser.addoption("--phenix-dir", action="store", default=None)
    parser.addoption("--phenix-labels-dir", action="store", default=None)
    parser.addoption("--czi-dir", action="store", default=None)
    parser.addoption("--sitetiff-dir", action="store", default=None)
    parser.addoption("--sitetiff-labels-dir", action="store", default=None)


# ---------------------------------------------------------------------------
# Data root fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_dir(request, tmp_path_factory):
    """Resolve the test data root directory.

    Returns a Path if data is available, None otherwise.
    Integration tests use the `requires_data` fixture which skips when None.
    """
    # 1. CLI option
    cli = request.config.getoption("--data-dir")
    if cli:
        p = Path(cli)
        if p.exists():
            return p
        pytest.fail(f"--data-dir path does not exist: {p}")

    # 2. Environment variable (local path)
    env = os.environ.get("TEST_DATA_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
        pytest.fail(f"TEST_DATA_DIR path does not exist: {p}")

    # 3. Environment variable (remote URL)
    url = os.environ.get("TEST_DATA_URL")
    if url:
        cache = tmp_path_factory.mktemp("testdata")
        return _download_test_data(url, cache)

    return None


def _download_test_data(base_url: str, cache_dir: Path) -> Path:
    """Download a test dataset from a public HTTP URL into cache_dir.

    The URL is expected to point to a manifest file (testdata.txt) that lists
    relative paths of files to download, one per line. Example manifest:

        imagexpress/plate1/t001_A01_s01_w1_z001.tif
        phenix/meas1/r01c01f01p01-ch1sk1fk1fl1.tif
        czi/experiment.czi

    Adjust this as needed when the Allas bucket structure is finalised.
    """
    manifest_url = base_url.rstrip("/") + "/testdata.txt"
    try:
        with urllib.request.urlopen(manifest_url) as resp:
            paths = resp.read().decode().splitlines()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"Could not fetch test data manifest from {manifest_url}: {exc}"
        )

    for rel_path in paths:
        rel_path = rel_path.strip()
        if not rel_path or rel_path.startswith("#"):
            continue
        dest = cache_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        file_url = base_url.rstrip("/") + "/" + rel_path
        urllib.request.urlretrieve(file_url, dest)

    return cache_dir


# ---------------------------------------------------------------------------
# Skip helper
# ---------------------------------------------------------------------------


@pytest.fixture
def requires_data(data_dir):
    """Fixture that skips the test if no test data is available."""
    if data_dir is None:
        pytest.skip(
            "No test data provided. "
            "Use --data-dir, TEST_DATA_DIR, or TEST_DATA_URL."
        )
    return data_dir


# ---------------------------------------------------------------------------
# Format-specific data fixtures (all skip gracefully if subdir is missing)
# ---------------------------------------------------------------------------


def _subdir(data_dir, name):
    if data_dir is None:
        return None
    p = data_dir / name
    return p if p.exists() else None


def _resolve_format_dir(request, cli_option, env_var, data_dir, subdir_name):
    """Resolve a format-specific directory: CLI > env var > data_dir subdir."""
    cli = request.config.getoption(cli_option)
    if cli:
        return Path(cli)
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    return _subdir(data_dir, subdir_name)


@pytest.fixture(scope="session")
def imagexpress_dir(request, data_dir):
    return _resolve_format_dir(
        request,
        "--imagexpress-dir",
        "TEST_DATA_IMAGEXPRESS",
        data_dir,
        "imagexpress",
    )


@pytest.fixture(scope="session")
def phenix_dir(request, data_dir):
    return _resolve_format_dir(
        request, "--phenix-dir", "TEST_DATA_PHENIX", data_dir, "phenix"
    )


@pytest.fixture(scope="session")
def phenix_labels_dir(request, data_dir):
    return _resolve_format_dir(
        request,
        "--phenix-labels-dir",
        "TEST_DATA_PHENIX_LABELS",
        data_dir,
        "phenix_labels",
    )


@pytest.fixture(scope="session")
def czi_dir(request, data_dir):
    return _resolve_format_dir(
        request, "--czi-dir", "TEST_DATA_CZI", data_dir, "czi"
    )


@pytest.fixture(scope="session")
def sitetiff_dir(request, data_dir):
    return _resolve_format_dir(
        request, "--sitetiff-dir", "TEST_DATA_SITETIFF", data_dir, "sitetiff"
    )


@pytest.fixture(scope="session")
def sitetiff_labels_dir(request, data_dir):
    return _resolve_format_dir(
        request,
        "--sitetiff-labels-dir",
        "TEST_DATA_SITETIFF_LABELS",
        data_dir,
        "sitetiff_labels",
    )


# ---------------------------------------------------------------------------
# Synthetic file fixtures for unit tests (no real image content needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def ix_files(tmp_path):
    """Empty files named according to ImageXpress convention."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    names = [
        "t001_A01_s1_w1_z001.tif",
        "t001_A01_s1_w2_z001.tif",
        "t001_A01_s2_w1_z001.tif",
        "t001_B02_s1_w1_z001.tif",
    ]
    files = []
    for name in names:
        f = plate / name
        f.touch()
        files.append(f)
    return files


@pytest.fixture
def phenix_files(tmp_path):
    """Empty files named according to Opera Phenix convention."""
    meas = tmp_path / "measurement1"
    meas.mkdir()
    names = [
        "r01c01f01p01-ch1sk1fk1fl1.tif",
        "r01c01f01p01-ch2sk1fk1fl1.tif",
        "r01c01f02p01-ch1sk1fk1fl1.tif",
        "r02c03f01p01-ch1sk1fk1fl1.tif",
    ]
    files = []
    for name in names:
        f = meas / name
        f.touch()
        files.append(f)
    return files


@pytest.fixture
def sitetiff_files(tmp_path):
    """Empty files named according to SiteTiff convention."""
    names = [
        "well_row1col1_site_001_DAPI.tif",
        "well_row1col1_site_002_DAPI.tif",
        "well_row2col3_site_001_DAPI.tif",
    ]
    files = []
    for name in names:
        f = tmp_path / name
        f.touch()
        files.append(f)
    return files


@pytest.fixture
def czi_file(tmp_path):
    """Empty file with .czi extension."""
    f = tmp_path / "experiment.czi"
    f.touch()
    return f
