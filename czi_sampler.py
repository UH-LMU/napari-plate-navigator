# czi_sampler.py
"""
Randomly sample scenes (and time frames) from a Zeiss .czi file and save them as TIFF images.

- Reading image data: bioio.BioImage
- Reading metadata (plate/well/site): czitools (best-effort; optional)
- Export: tifffile
- Usable as an importable module (functions) and as a CLI via click.

Notes
-----
This is a draft implementation meant for testing and iteration.
The czitools API can vary between versions; the helper that maps scene -> (well, site)
tries a few common patterns and falls back to minimal labels if unavailable.

Requirements (install via pip)
------------------------------
    pip install bioio czitools tifffile click numpy

Example (CLI)
-------------
    python czi_sampler.py input.czi --output-dir ./samples --n-images 12 --n-time-frames 3 --channels all
    python czi_sampler.py input.czi -o ./samples -n 8 -t 5 --channels 0,1 --seed 42

Example (Notebook)
------------------
    from czi_sampler import sample_czi_to_tiffs, list_scenes
    samples = sample_czi_to_tiffs(
        input_czi="input.czi",
        output_dir="./samples",
        n_images=10,
        n_time_frames=3,
        channels="all",
        seed=1,
        overwrite=True,
    )
    samples

"""
from __future__ import annotations

import os
import re
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

try:
    from bioio import BioImage
except Exception as e:
    raise ImportError(
        "bioio is required. Install with `pip install bioio`. Original error: %s" % e
    )

# czitools is optional; we handle absence gracefully
try:
    import czitools  # type: ignore
except Exception:
    czitools = None  # type: ignore

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SceneInfo:
    scene_id: Union[int, str]
    well: Optional[str] = None
    site: Optional[Union[int, str]] = None
    description: Optional[str] = None

# -----------------------------
# Metadata helpers (czitools)
# -----------------------------

def _parse_scene_infos_with_czitools(input_czi: str) -> Dict[Union[int, str], SceneInfo]:
    """
    Best-effort extraction of scene -> (well, site, description) using czitools.
    Returns a dict keyed by scene id (int or str) mapped to SceneInfo.

    This function attempts multiple czitools patterns to remain compatible across versions.
    If czitools is unavailable or parsing fails, returns an empty dict.
    """
    results: Dict[Union[int, str], SceneInfo] = {}
    if czitools is None:
        return results

    try:
        # Pattern 1: czitools provides a high-level reader that exposes metadata XML
        # Many czitools variants expose something like `czitools.read_metadata(input_czi)`
        md = None
        if hasattr(czitools, "read_metadata"):
            try:
                md = czitools.read_metadata(input_czi)  # returns dict-like or object
            except Exception:
                md = None
        # Pattern 2: a class exposing .meta or .metadata_xml
        if md is None:
            reader = None
            for attr in ("CziFile", "CZIFile", "Reader", "CZIReader"):
                if hasattr(czitools, attr):
                    try:
                        reader = getattr(czitools, attr)(input_czi)
                        break
                    except Exception:
                        reader = None
            if reader is not None:
                # Try common attributes
                if hasattr(reader, "meta"):
                    md = reader.meta  # xml string or parsed tree
                elif hasattr(reader, "metadata"):
                    md = reader.metadata
                elif hasattr(reader, "metadata_xml"):
                    md = reader.metadata_xml

        # Now extract wells/sites heuristically from XML or dict
        if md is None:
            return results

        # Normalize to string if possible
        md_str = None
        try:
            if isinstance(md, str):
                md_str = md
            else:
                md_str = json.dumps(md)
        except Exception:
            md_str = None

        if md_str:
            # Heuristic regex: look for Well IDs like "A01", "B12" and Site numbers
            # Also attempt to detect Scene indices near those fields.
            # This is intentionally permissive; adjust as needed for your czitools version.
            well_pattern = re.compile(r"Well(?:ID|Name)?\"?\s*[:=]\s*\"?([A-H]\d{2})", re.IGNORECASE)
            site_pattern = re.compile(r"Site\"?\s*[:=]\s*\"?(\d+)")
            # Scenes often appear as Index or Scene/ID entries
            scene_int_pattern = re.compile(r"Scene(?:Index|ID)\"?\s*[:=]\s*\"?(\d+)")
            scene_str_pattern = re.compile(r"Scene\"?\s*[:=]\s*\"?([A-Za-z0-9_-]+)")

            wells = well_pattern.findall(md_str)
            sites = site_pattern.findall(md_str)
            scene_ints = [int(s) for s in scene_int_pattern.findall(md_str)]
            scene_strs = scene_str_pattern.findall(md_str)

            # Build scene info list; if counts differ, we pair by min length
            n_candidates = max(len(scene_ints), len(scene_strs), len(wells), len(sites))
            for i in range(n_candidates):
                scene_id: Union[int, str]
                if i < len(scene_ints):
                    scene_id = scene_ints[i]
                elif i < len(scene_strs):
                    scene_id = scene_strs[i]
                else:
                    scene_id = i  # fallback index
                well = wells[i] if i < len(wells) else None
                site = sites[i] if i < len(sites) else None
                results[scene_id] = SceneInfo(scene_id=scene_id, well=well, site=site)
        else:
            # If md is a dict-like, try common nested keys
            try:
                def get(d, *keys):
                    cur = d
                    for k in keys:
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            return None
                    return cur
                scenes = get(md, "Information", "Image", "S", "Scenes") or []
                for i, sc in enumerate(scenes):
                    well = get(sc, "Well") or get(sc, "WellID")
                    site = get(sc, "Site")
                    results[i] = SceneInfo(scene_id=i, well=well, site=site)
            except Exception:
                pass
    except Exception:
        # Swallow errors; we'll proceed without scene mapping.
        return {}

    return results

# -----------------------------
# BioImage helpers
# -----------------------------

def list_scenes(input_czi: str) -> List[Union[int, str]]:
    """Return a list of scene identifiers from the CZI using bioio.BioImage."""
    img = BioImage(input_czi)
    scenes = []
    # BioImage.scenes typically provides scene names; fallback to range(n_scenes)
    if hasattr(img, "scenes") and isinstance(img.scenes, (list, tuple)) and len(img.scenes) > 0:
        scenes = list(img.scenes)
    else:
        # Try attribute .scene_count or infer via set_scene iteration
        if hasattr(img, "set_scene"):
            i = 0
            while True:
                try:
                    img.set_scene(i)
                    scenes.append(i)
                    i += 1
                except Exception:
                    break
    return scenes


def _get_time_length(img: BioImage) -> int:
    """Return T length for current scene; 0/1 if no time dimension."""
    # Try using dims metadata if available
    for probe in ("T", "t"):
        try:
            # Some BioImage versions expose `dims` as dict-like with axis lengths
            if hasattr(img, "dims") and probe in getattr(img, "dims"):
                t_len = int(img.dims[probe])
                return t_len
        except Exception:
            pass
    # Fallback: attempt to read minimal data to infer T by iterating T indices
    # We try get_image_data with T index incrementally until failure.
    t_len = 0
    if hasattr(img, "get_image_data"):
        try:
            # Probe time indices up to a reasonable bound (avoid loading full data)
            for ti in range(0, 4096):
                try:
                    _ = img.get_image_data("CZYX", T=ti)
                    t_len += 1
                except Exception:
                    break
        except Exception:
            pass
    # If zero but get_image_data without T succeeds, assume 1
    if t_len == 0:
        try:
            _ = img.get_image_data("CZYX")
            t_len = 1
        except Exception:
            t_len = 0
    return t_len


def _get_channel_count(img: BioImage) -> int:
    """Return channel count for current scene."""
    # Attempt via channel_names
    if hasattr(img, "channel_names") and img.channel_names:
        return len(img.channel_names)
    # Probe by trying C indices
    c_len = 0
    if hasattr(img, "get_image_data"):
        try:
            for ci in range(0, 1024):
                try:
                    _ = img.get_image_data("ZYX", C=ci)  # request single channel
                    c_len += 1
                except Exception:
                    break
        except Exception:
            pass
    if c_len == 0:
        try:
            arr = img.get_image_data("CZYX")
            c_len = arr.shape[0]
        except Exception:
            c_len = 1
    return c_len


def _read_frame(img: BioImage, t_index: Optional[int], channels: Optional[Sequence[int]]) -> np.ndarray:
    """
    Read a single time frame as a numpy array with axes CZYX.
    If channels is provided, subset channels accordingly.
    """
    kwargs = {}
    if t_index is not None:
        kwargs["T"] = int(t_index)
    if channels is not None:
        kwargs["C"] = list(channels)
    arr = img.get_image_data("CZYX", **kwargs)
    # Ensure numpy array
    if hasattr(arr, "compute"):
        arr = arr.compute()
    return np.asarray(arr)

# -----------------------------
# Sampling core
# -----------------------------

def sample_czi_to_tiffs(
    input_czi: str,
    output_dir: str,
    n_images: int = 10,
    n_time_frames: int = 3,
    channels: Union[str, Sequence[int]] = "all",
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> List[str]:
    """
    Randomly sample scenes (and T frames if present) from a .czi and save TIFFs.

    Parameters
    ----------
    input_czi : str
        Path to the input .czi file.
    output_dir : str
        Directory where TIFFs will be written (created if missing).
    n_images : int, default 10
        Number of samples to export. Samples are taken across the Scene axis.
    n_time_frames : int, default 3
        If the data has a time-lapse (T>1), include this many random T frames per sample.
        If T<=1, a single frame is exported.
    channels : "all" or sequence of ints
        Channels to include. Use "all" to include all channels. Otherwise pass indices like [0, 1].
    seed : int, optional
        Random seed for reproducibility.
    overwrite : bool, default False
        Overwrite existing files if they exist.

    Returns
    -------
    List[str]
        Paths to the written TIFF files.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Resolve scene mapping via czitools (optional)
    scene_map = _parse_scene_infos_with_czitools(input_czi)

    # Load BioImage and list scenes
    img = BioImage(input_czi)
    scene_ids = list_scenes(input_czi)
    if len(scene_ids) == 0:
        # Fallback: assume single scene index 0
        scene_ids = [0]

    # Sample scene indices (without replacement)
    k = min(n_images, len(scene_ids))
    chosen_scenes = random.sample(scene_ids, k)

    written_paths: List[str] = []

    for s in chosen_scenes:
        # Select scene
        try:
            img.set_scene(s)
            scene_label = str(s)
        except Exception:
            # If set_scene fails for non-int scene IDs, try index lookup
            scene_label = str(s)
            if hasattr(img, "scenes") and s in img.scenes:
                try:
                    img.set_scene(img.scenes.index(s))
                except Exception:
                    # Can't set; proceed assuming already set/default
                    pass

        # Determine time frames
        t_len = _get_time_length(img)
        if t_len is None or t_len <= 1:
            t_indices = [None]  # single frame (no time dimension)
        else:
            # Pick distinct random t indices
            k_t = min(n_time_frames, t_len)
            t_indices = sorted(random.sample(list(range(t_len)), k_t))

        # Determine channels
        channel_indices: Optional[Sequence[int]]
        if isinstance(channels, str) and channels.lower() == "all":
            channel_indices = None  # include all
        else:
            channel_indices = list(map(int, channels))  # type: ignore[arg-type]

        # Read each selected time frame (CZYX)
        frames: List[np.ndarray] = []
        for ti in t_indices:
            frame = _read_frame(img, t_index=ti, channels=channel_indices)
            # Ensure dtype is something common (e.g., uint16) to avoid large files
            # We'll keep original dtype to preserve data unless it's float64
            if frame.dtype == np.float64:
                frame = frame.astype(np.float32)
            frames.append(frame)

        # Stack frames along a new leading T axis
        if len(frames) == 1:
            stack = frames[0][None, ...]  # shape (1, C, Z, Y, X)
        else:
            stack = np.stack(frames, axis=0)  # shape (T, C, Z, Y, X)

        # Build filename with scene, well, site, and time indices
        info = scene_map.get(s) or scene_map.get(str(s))
        well = (info.well if info else None) or "well"
        site = (info.site if info else None) or "site"
        t_label = (
            "t" + "-".join(["all" if ti is None else str(ti) for ti in t_indices])
        )
        base = f"scene-{scene_label}_{well}-{site}_{t_label}.tif"
        out_path = os.path.join(output_dir, base)

        if os.path.exists(out_path) and not overwrite:
            # Create a unique suffix
            suf = 1
            while True:
                candidate = os.path.join(output_dir, f"scene-{scene_label}_{well}-{site}_{t_label}_{suf}.tif")
                if not os.path.exists(candidate):
                    out_path = candidate
                    break
                suf += 1

        # Save TIFF with axes metadata (so downstream knows dimension order)
        # Axes string: TCZYX
        tifffile.imwrite(
            out_path,
            stack,
            photometric="minisblack",  # multi-channel will be planar-separate
            metadata={"axes": "TCZYX"},
        )
        written_paths.append(out_path)

    return written_paths


# -----------------------------
# CLI
# -----------------------------

import click

@click.command()
@click.argument("input_czi", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--output-dir", "output_dir", type=click.Path(file_okay=False, path_type=str), default="czi_samples", help="Directory to write sampled TIFFs.")
@click.option("--n-images", "n_images", type=int, default=10, show_default=True, help="Number of samples (scenes) to export.")
@click.option("--n-time-frames", "n_time_frames", type=int, default=3, show_default=True, help="Number of time frames per sample if time-lapse is present.")
@click.option("--channels", "channels", type=str, default="all", show_default=True, help="Channels to include: 'all' or comma-separated indices, e.g. '0,1'.")
@click.option("--seed", "seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--overwrite/--no-overwrite", "overwrite", default=False, show_default=True, help="Overwrite existing files.")
def main(input_czi: str, output_dir: str, n_images: int, n_time_frames: int, channels: str, seed: Optional[int], overwrite: bool) -> None:
    """CLI entry point."""
    # Parse channels option
    ch: Union[str, Sequence[int]]
    if isinstance(channels, str) and channels.lower().strip() == "all":
        ch = "all"
    else:
        try:
            ch = [int(x.strip()) for x in channels.split(",") if x.strip() != ""]
        except Exception:
            raise click.BadOptionUsage("channels", "Invalid channels specification. Use 'all' or comma-separated indices like '0,1'.")

    written = sample_czi_to_tiffs(
        input_czi=input_czi,
        output_dir=output_dir,
        n_images=n_images,
        n_time_frames=n_time_frames,
        channels=ch,
        seed=seed,
        overwrite=overwrite,
    )
    click.echo(f"Wrote {len(written)} TIFFs to {output_dir}")
    for p in written:
        click.echo(f"- {p}")


if __name__ == "__main__":
    main()
