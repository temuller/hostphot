"""Herschel/HSA image discovery and download helpers.

This module follows HostPhot's cutout interface: ``download_images(...,
survey="Herschel")`` calls ``get_Herschel_images`` and expects one HDUList per
requested filter.  The HSA products are full processed maps rather than cutouts,
so the selected maps are cut to the requested ``size`` after the download.

Only the photometry observations whose footprint contains the target are used, and
only the final ("standalone") maps of each observation are downloaded (the whole
processing level is downloaded only if these are not available).
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astroquery.esa.hsa import HSA

from astropy.wcs import WCS

from hostphot.surveys_utils import check_filters_validity, get_survey_filters
from hostphot.utils import image_coverage, select_image, FULL_COVERAGE


DEFAULT_FILTERS = ("PACS100", "PACS160", "SPIRE250", "SPIRE350", "SPIRE500")
DEFAULT_INSTRUMENTS = ("PACS", "SPIRE")
DEFAULT_PRODUCT_LEVELS = ("LEVEL2_5", "LEVEL3", "LEVEL2")
DEFAULT_RADIUS_ARCMIN = 5.0

INSTRUMENT_OID_TO_NAME = {1: "PACS", 2: "SPIRE", 3: "HIFI"}

# photometry observing modes (hsa.observing_mode): PacsPhoto, SpirePhotoLargeScan,
# SpirePhotoPointJiggle, SpirePhotoSmall, SpirePhotoPointJiggleId55, SpirePhotoSmallScan
# and SpirePacsParallel (with both PACS and SPIRE maps)
PHOTOMETRY_MODES = {1: ("PACS",), 4: ("SPIRE",), 5: ("SPIRE",), 6: ("SPIRE",), 7: ("SPIRE",),
                    283: ("SPIRE",), 31: ("PACS", "SPIRE")}

# Host-galaxy photometry should prefer extended-emission products.
FILTER_RULES = {
    "PACS100": {
        "instrument": "PACS",
        "wavelength": 100.0,
        "tokens": ("HPPJSMAPB", "HPPUNIMAPB", "HPPHPFMAPB"),
    },
    "PACS160": {
        "instrument": "PACS",
        "wavelength": 160.0,
        "tokens": ("HPPJSMAPR", "HPPUNIMAPR", "HPPHPFMAPR"),
    },
    "SPIRE250": {
        "instrument": "SPIRE",
        "wavelength": 250.0,
        "tokens": ("extdPSW",),
    },
    "SPIRE350": {
        "instrument": "SPIRE",
        "wavelength": 350.0,
        "tokens": ("extdPMW",),
    },
    "SPIRE500": {
        "instrument": "SPIRE",
        "wavelength": 500.0,
        "tokens": ("extdPLW",),
    },
}

BAD_PRODUCT_TOKENS = (
    "context",
    "diag",
    "diagnostic",
    "quality",
    "coverage",
    "mask",
    "flag",
)


def _normalise_filters(filters: Optional[str | Iterable[str]]) -> list[str]:
    """Return filters as a list; Herschel filter names are multi-character."""
    if filters is None:
        filters = get_survey_filters("Herschel")
    elif isinstance(filters, str):
        filters = [filters]
    filters = list(filters)
    check_filters_validity(filters, "Herschel")
    return filters


def query_hsa_observations(
    ra: float,
    dec: float,
    radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
    instruments: Iterable[str] = DEFAULT_INSTRUMENTS,
    max_obs: int = 200,
) -> pd.DataFrame:
    """Query the public HSA photometry observations whose footprint contains RA/Dec.

    Spectroscopy, calibration and engineering observations are excluded. Observations
    in SPIRE/PACS parallel mode are listed once per instrument. The observations are
    sorted by duration (longest, i.e. deepest, first).

    ``radius_arcmin`` is kept for backward compatibility, but it is not used (the
    footprints of the observations are used instead of the distance to their centres).
    """
    del radius_arcmin
    modes = ",".join(str(mode) for mode in PHOTOMETRY_MODES)
    query = (
        f"select top {max_obs} observation_id, instrument_oid, observing_mode_oid, target_name, "
        "ra, dec, duration from hsa.v_active_observation "
        f"where contains(point('ICRS', {ra}, {dec}), polygon_fov)=1 "
        f"and observing_mode_oid in ({modes})"
    )
    df = HSA.query_hsa_tap(query).to_pandas()
    if df.empty:
        return df

    keep = {inst.upper() for inst in instruments}
    rows = []
    for _, row in df.iterrows():
        for instrument in PHOTOMETRY_MODES[int(row["observing_mode_oid"])]:
            if instrument in keep:
                rows.append({**row.to_dict(), "instrument_name": instrument})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("duration", ascending=False, kind="stable")
    order = ["observation_id", "instrument_name", "instrument_oid", "target_name", "ra", "dec"]
    order += [col for col in df.columns if col not in order]
    return df[order].reset_index(drop=True)


def _download_product(
    observation_id: str | int,
    instrument_name: str,
    product_level: Optional[str],
    download_dir: Path,
    overwrite: bool = False,
) -> Optional[Path]:
    """Download one compressed HSA product tarball.

    If ``product_level`` is ``None``, only the final ("standalone") maps of the
    observation are downloaded; otherwise, the whole processing level.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    if product_level is None:
        product_level = "STANDALONE"
    stem = f"Herschel_{instrument_name}_{observation_id}_{product_level}"
    existing = list(download_dir.glob(stem + ".*"))
    if existing and not overwrite:
        return existing[0]

    try:
        level_kwargs = {} if product_level == "STANDALONE" else {"product_level": product_level}
        filename = HSA.download_data(
            retrieval_type="STANDALONE" if product_level == "STANDALONE" else "OBSERVATION",
            observation_id=str(observation_id),
            instrument_name=instrument_name,
            **level_kwargs,
            filename=stem,
            download_dir=str(download_dir),
            cache=False,
            compress="true",
            verbose=False,
        )
    except Exception as exc:
        if "proprietary" in str(exc).lower():
            raise PermissionError(f"Herschel observation {observation_id} is not public") from exc
        print(
            "  Herschel download failed: "
            f"obs={observation_id} instrument={instrument_name} level={product_level}: {exc}"
        )
        return None
    return Path(filename)


def _safe_extract_tarball(tarball: Path, extract_dir: Path, overwrite: bool = False) -> Path:
    """Extract a tarball while rejecting paths outside the target directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists() and not overwrite:
        return extract_dir

    target = extract_dir.resolve()
    with tarfile.open(tarball, "r:*") as tf:
        for member in tf.getmembers():
            member_path = (extract_dir / member.name).resolve()
            if target not in member_path.parents and member_path != target:
                raise RuntimeError(f"Unsafe path in HSA tarball: {member.name}")
        tf.extractall(extract_dir)
    marker.write_text(str(tarball) + "\n")
    return extract_dir


def _image_hdu_index(hdul: fits.HDUList) -> Optional[int]:
    """Find the first 2D image extension in an HSA FITS product."""
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and int(hdu.header.get("NAXIS", 0) or 0) >= 2:
            return index
    return None


def _candidate_filter(path: Path, header: fits.Header) -> Optional[str]:
    """Infer the HostPhot Herschel filter from filename/header metadata."""
    path_text = str(path).lower()
    wavelength = header.get("WAVELNTH")
    instrument = str(header.get("INSTRUME", "")).upper()

    for filt, rule in FILTER_RULES.items():
        if rule["instrument"] not in instrument:
            continue
        if any(token.lower() in path_text for token in rule["tokens"]):
            return filt
        if wavelength is not None and abs(float(wavelength) - rule["wavelength"]) < 1.0:
            return filt
    return None


def _score_candidate(path: Path, filt: str, header: fits.Header) -> int:
    """Rank competing products for a filter."""
    text = str(path).lower()
    if any(token in text for token in BAD_PRODUCT_TOKENS):
        return -100

    rule = FILTER_RULES[filt]
    score = 0
    for rank, token in enumerate(rule["tokens"]):
        if token.lower() in text:
            score += 100 - rank * 10
            break

    bunit = str(header.get("BUNIT", "")).lower()
    if filt.startswith("PACS") and bunit == "jy/pixel":
        score += 20
    if filt.startswith("SPIRE") and bunit == "mjy/sr":
        score += 20

    wavelength = header.get("WAVELNTH")
    if wavelength is not None and abs(float(wavelength) - rule["wavelength"]) < 1.0:
        score += 10
    return score


def _scan_science_maps(extract_dir: Path, requested_filters: Iterable[str]) -> pd.DataFrame:
    """Find and rank science-map FITS products in an extracted HSA tree."""
    requested = set(requested_filters)
    rows = []
    for path in list(extract_dir.rglob("*.fits")) + list(extract_dir.rglob("*.fits.gz")):
        try:
            with fits.open(path, memmap=False) as hdul:
                image_index = _image_hdu_index(hdul)
                if image_index is None:
                    continue
                image_header = hdul[image_index].header
                merged_header = hdul[0].header.copy()
                merged_header.update(image_header)
                filt = _candidate_filter(path, merged_header)
                if filt not in requested:
                    continue
                score = _score_candidate(path, filt, merged_header)
                if score <= 0:
                    continue
                rows.append(
                    {
                        "filter": filt,
                        "path": str(path),
                        "score": score,
                        "image_ext": image_index,
                        "instrument": merged_header.get("INSTRUME"),
                        "wavelength": merged_header.get("WAVELNTH"),
                        "bunit": merged_header.get("BUNIT"),
                        "naxis1": merged_header.get("NAXIS1"),
                        "naxis2": merged_header.get("NAXIS2"),
                    }
                )
        except Exception as exc:
            print(f"  could not inspect Herschel FITS {path}: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["filter", "score", "path"], ascending=[True, False, True])


def _make_hostphot_hdul(path: Path, image_ext: int, filt: str) -> fits.HDUList:
    """Return a HostPhot-ready HDUList with science image in the primary HDU."""
    with fits.open(path, memmap=False) as hdul:
        image_hdu = hdul[image_ext]
        header = image_hdu.header.copy()
        header.update(
            {
                "ORIGIN": "Herschel Science Archive",
                "SURVEY": "Herschel",
                "FILTER": filt,
                "SRCFILE": path.name[:68],
                "SRCEXT": image_ext,
            }
        )
        for key in ("TELESCOP", "INSTRUME", "OBJECT", "OBS_ID", "WAVELNTH"):
            if key in hdul[0].header and key not in header:
                header[key] = hdul[0].header[key]
        data = image_hdu.data.copy()
    return fits.HDUList([fits.PrimaryHDU(data=data, header=header)])


def _cutout(hdul: fits.HDUList, ra: float, dec: float, size_arcsec: float) -> fits.HDUList:
    """Cuts a map (in the primary HDU) to a square of the given size centred on RA/Dec."""
    header = hdul[0].header
    img_wcs = WCS(header, naxis=2)
    pixel_scale = float(np.sqrt(np.abs(np.linalg.det(img_wcs.pixel_scale_matrix))) * 3600)
    size_pixels = max(int(round(size_arcsec / pixel_scale)), 1)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cutout = Cutout2D(hdul[0].data, coords, size_pixels, wcs=img_wcs)
    header = header.copy()
    header.update(cutout.wcs.to_header())
    return fits.HDUList([fits.PrimaryHDU(data=cutout.data, header=header)])


def get_Herschel_images(
    ra: float,
    dec: float,
    size: float | u.Quantity = 3,
    filters: Optional[str | Iterable[str]] = None,
    output_dir: Optional[str | Path] = None,
    radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
    product_levels: Iterable[str] = DEFAULT_PRODUCT_LEVELS,
    overwrite: bool = False,
    max_obs: int = 200,
) -> list[Optional[fits.HDUList]]:
    """Download Herschel maps and return one HostPhot-ready HDUList per filter.

    The photometry observations whose footprint contains the target are downloaded one
    by one, from the longest (deepest) to the shortest, until a map covering the whole
    requested area is found for each filter. Only the final ("standalone") maps of each
    observation are downloaded; the processing levels in ``product_levels`` are only
    downloaded (whole) if these do not include the filter. Among the maps of a filter,
    the deepest one covering the whole area is used; otherwise, the one with the largest
    coverage (with a warning).

    Parameters
    ----------
    ra, dec:
        Target coordinates in degrees.
    size:
        Image size. If a float is given, the units are assumed to be arcmin. The maps are
        downloaded whole (HSA does not provide cutouts) and cut to this size.
    filters:
        Herschel filters to return. Valid options are ``PACS100``, ``PACS160``,
        ``SPIRE250``, ``SPIRE350`` and ``SPIRE500``.
    output_dir:
        Optional HostPhot survey folder used to store CSV provenance files.
    radius_arcmin:
        Not used (kept for backward compatibility; see :func:`query_hsa_observations`).
    product_levels:
        HSA product levels to try, in order, if the final maps do not include a filter.
    overwrite:
        Redownload and re-extract products even when local files exist.
    max_obs:
        Maximum number of HSA observations returned by the query.
    """
    requested_filters = _normalise_filters(filters)
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    if output_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="hostphot_herschel_")
        target_dir = Path(tmp.name)
    else:
        tmp = None
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = target_dir / "hsa_downloads"
    extracted_dir = target_dir / "hsa_extracted"

    instruments = {FILTER_RULES[filt]["instrument"] for filt in requested_filters}
    observations = query_hsa_observations(ra, dec, instruments=instruments, max_obs=max_obs)
    observations.to_csv(target_dir / "hsa_observations.csv", index=False)
    if observations.empty:
        if tmp is not None:
            tmp.cleanup()
        return [None for _ in requested_filters]

    downloads_log = []
    # maps of each filter: (data, wcs, depth) for the selection, and their provenance
    maps: dict[str, list] = {filt: [] for filt in requested_filters}
    hduls: dict[str, list] = {filt: [] for filt in requested_filters}
    provenance: dict[str, list] = {filt: [] for filt in requested_filters}
    covered: set[str] = set()

    def add_maps(candidates: pd.DataFrame, obs_id, instrument: str, level: str, duration: float) -> set:
        """Adds the best map of each filter in an observation; returns the filters found."""
        found = set()
        for filt, filt_df in candidates.groupby("filter"):
            if filt in covered or filt not in maps:
                continue
            best = filt_df.sort_values(["score", "path"], ascending=[False, True]).iloc[0]
            hdul = _make_hostphot_hdul(Path(best["path"]), int(best["image_ext"]), filt)
            img_wcs = WCS(hdul[0].header, naxis=2)
            maps[filt].append((hdul[0].data, img_wcs, duration))
            hduls[filt].append(hdul)
            coverage = image_coverage(hdul[0].data, img_wcs, ra, dec, size_arcsec)
            provenance[filt].append(
                {**best.to_dict(), "observation_id": obs_id, "instrument_name": instrument,
                 "product_level": level, "duration": duration, "coverage": coverage}
            )
            found.add(filt)
            if coverage >= FULL_COVERAGE:
                covered.add(filt)
        return found

    for _, row in observations.iterrows():
        instrument = row["instrument_name"]
        pending = [
            filt for filt in requested_filters
            if filt not in covered and FILTER_RULES[filt]["instrument"] == instrument
        ]
        if len(pending) == 0:
            continue
        obs_id, duration = row["observation_id"], float(row["duration"])
        # final maps first; whole processing levels only if they do not include the filters
        for level in [None] + list(product_levels):
            level_name = level or "STANDALONE"
            try:
                tarball = _download_product(obs_id, instrument, level, downloads_dir, overwrite=overwrite)
            except PermissionError:
                break  # proprietary data (no level can be downloaded)
            if tarball is None or not tarball.exists() or tarball.stat().st_size == 0:
                continue
            extract_dir = extracted_dir / f"{instrument}_{obs_id}_{level_name}"
            _safe_extract_tarball(tarball, extract_dir, overwrite=overwrite)
            candidates = _scan_science_maps(extract_dir, pending)
            downloads_log.append(
                {"observation_id": obs_id, "instrument": instrument, "product_level": level_name,
                 "tarball": str(tarball), "n_candidates": len(candidates)}
            )
            found = add_maps(candidates, obs_id, instrument, level_name, duration) if not candidates.empty else set()
            pending = [filt for filt in pending if filt not in found]
            if len(pending) == 0:
                break
        if covered.issuperset(requested_filters):
            break  # all the filters have a map covering the whole area

    if downloads_log:
        pd.DataFrame(downloads_log).to_csv(target_dir / "hsa_downloads.csv", index=False)
    all_provenance = [entry for filt in requested_filters for entry in provenance[filt]]
    if all_provenance:
        pd.DataFrame(all_provenance).to_csv(target_dir / "hsa_candidate_maps.csv", index=False)

    result = []
    selected = []
    for filt in requested_filters:
        index = select_image(maps[filt], ra, dec, size_arcsec, label=f"Herschel {filt}")
        if index is None:
            result.append(None)
            continue
        result.append(_cutout(hduls[filt][index], ra, dec, size_arcsec))
        selected.append(provenance[filt][index])
    if selected:
        pd.DataFrame(selected).to_csv(target_dir / "hsa_selected_maps.csv", index=False)

    # Delete temporary files to save space
    import shutil
    if downloads_dir.exists():
        shutil.rmtree(downloads_dir)
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    if tmp is not None:
        tmp.cleanup()
    return result
