"""Herschel/HSA image discovery and download helpers.

This module follows HostPhot's cutout interface: ``download_images(...,
survey="Herschel")`` calls ``get_Herschel_images`` and expects one HDUList per
requested filter.  The HSA products are full processed maps rather than cutouts,
so the ``size`` argument is accepted for API compatibility but is not used.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.esa.hsa import HSA

from hostphot.surveys_utils import check_filters_validity, get_survey_filters


DEFAULT_FILTERS = ("PACS100", "PACS160", "SPIRE250", "SPIRE350", "SPIRE500")
DEFAULT_INSTRUMENTS = ("PACS", "SPIRE")
DEFAULT_PRODUCT_LEVELS = ("LEVEL2_5", "LEVEL3", "LEVEL2")
DEFAULT_RADIUS_ARCMIN = 5.0

INSTRUMENT_OID_TO_NAME = {1: "PACS", 2: "SPIRE", 3: "HIFI"}

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
    """Query public HSA observations whose pointing centre is near RA/Dec."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    table = HSA.query_region(coord, radius_arcmin * u.arcmin, n_obs=max_obs, columns="*")
    df = table.to_pandas()
    if df.empty:
        return df

    if "observation_id" not in df.columns:
        raise RuntimeError(f"HSA result has no observation_id column: {list(df.columns)}")

    inst_col = next((col for col in ("instrument_name", "instrument") if col in df.columns), None)
    if inst_col is not None:
        df["instrument_name"] = df[inst_col].astype(str).str.upper()
    elif "instrument_oid" in df.columns:
        df["instrument_name"] = pd.to_numeric(df["instrument_oid"], errors="coerce").map(
            INSTRUMENT_OID_TO_NAME
        )
    else:
        raise RuntimeError(
            "HSA result has no instrument_name/instrument or instrument_oid column: "
            f"{list(df.columns)}"
        )

    keep = {inst.upper() for inst in instruments}
    df = df[df["instrument_name"].isin(keep)].copy()
    order = ["observation_id", "instrument_name", "instrument_oid", "target_name", "ra", "dec"]
    order = [col for col in order if col in df.columns]
    order += [col for col in df.columns if col not in order]
    return df[order].reset_index(drop=True)


def _download_product(
    observation_id: str | int,
    instrument_name: str,
    product_level: str,
    download_dir: Path,
    overwrite: bool = False,
) -> Optional[Path]:
    """Download one compressed HSA product tarball."""
    download_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Herschel_{instrument_name}_{observation_id}_{product_level}"
    existing = list(download_dir.glob(stem + ".*"))
    if existing and not overwrite:
        return existing[0]

    try:
        filename = HSA.download_data(
            retrieval_type="OBSERVATION",
            observation_id=str(observation_id),
            instrument_name=instrument_name,
            product_level=product_level,
            filename=stem,
            download_dir=str(download_dir),
            cache=False,
            compress="true",
            verbose=False,
        )
    except Exception as exc:
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

    Parameters
    ----------
    ra, dec:
        Target coordinates in degrees.
    size:
        Accepted only for HostPhot API compatibility. HSA returns processed maps,
        not on-demand cutouts.
    filters:
        Herschel filters to return. Valid options are ``PACS100``, ``PACS160``,
        ``SPIRE250``, ``SPIRE350`` and ``SPIRE500``.
    output_dir:
        Optional HostPhot survey folder used to store HSA tarballs, extracted
        products and CSV provenance files.
    radius_arcmin:
        HSA observation search radius.
    product_levels:
        HSA product levels to try, in order.
    overwrite:
        Redownload and re-extract products even when local files exist.
    max_obs:
        Maximum number of HSA observations returned by the cone search.
    """
    del size  # HSA products are not cutout-size driven.
    requested_filters = _normalise_filters(filters)

    if output_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="hostphot_herschel_")
        target_dir = Path(tmp.name)
    else:
        tmp = None
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = target_dir / "hsa_downloads"
    extracted_dir = target_dir / "hsa_extracted"

    observations = query_hsa_observations(ra, dec, radius_arcmin=radius_arcmin, max_obs=max_obs)
    observations.to_csv(target_dir / "hsa_observations.csv", index=False)
    if observations.empty:
        if tmp is not None:
            tmp.cleanup()
        return [None for _ in requested_filters]

    all_candidates = []
    downloads_log = []
    selected_by_filter: dict[str, dict] = {}

    for _, row in observations.iterrows():
        obs_id = row["observation_id"]
        instrument = str(row["instrument_name"]).upper()
        if instrument not in DEFAULT_INSTRUMENTS:
            continue

        for level in product_levels:
            tarball = _download_product(obs_id, instrument, level, downloads_dir, overwrite=overwrite)
            if tarball is None or not tarball.exists() or tarball.stat().st_size == 0:
                continue

            extract_dir = extracted_dir / f"{instrument}_{obs_id}_{level}"
            _safe_extract_tarball(tarball, extract_dir, overwrite=overwrite)
            candidates = _scan_science_maps(extract_dir, requested_filters)
            downloads_log.append(
                {
                    "observation_id": obs_id,
                    "instrument": instrument,
                    "product_level": level,
                    "tarball": str(tarball),
                    "n_candidates": len(candidates),
                }
            )
            if not candidates.empty:
                candidates["observation_id"] = obs_id
                candidates["product_level"] = level
                all_candidates.append(candidates)

            # Stop after the first product level that actually exists for this observation.
            break

    if downloads_log:
        pd.DataFrame(downloads_log).to_csv(target_dir / "hsa_downloads.csv", index=False)

    if all_candidates:
        candidates_df = pd.concat(all_candidates, ignore_index=True)
        candidates_df.to_csv(target_dir / "hsa_candidate_maps.csv", index=False)
        for filt in requested_filters:
            filt_df = candidates_df[candidates_df["filter"] == filt]
            if filt_df.empty:
                continue
            selected_by_filter[filt] = filt_df.sort_values(
                ["score", "product_level", "path"], ascending=[False, True, True]
            ).iloc[0].to_dict()

    if selected_by_filter:
        pd.DataFrame(selected_by_filter.values()).to_csv(target_dir / "hsa_selected_maps.csv", index=False)

    result = []
    for filt in requested_filters:
        selected = selected_by_filter.get(filt)
        if selected is None:
            result.append(None)
            continue
        result.append(_make_hostphot_hdul(Path(selected["path"]), int(selected["image_ext"]), filt))

    if tmp is not None:
        tmp.cleanup()
    return result
