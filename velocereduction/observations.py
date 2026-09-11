"""Identify and classify Veloce observations for reduction."""

from calendar import month_abbr
from pathlib import Path
import logging
import shutil
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .constants import CALIBRATION_EXPTIMES, CALIBRATION_TYPES, EXPECTED_CCDS

logger = logging.getLogger(__name__)


def raw_fits_path(paths, night, run, ccd):
    """Return the expected raw FITS path for a run and CCD."""
    day, month = night[-2:], month_abbr[int(night[-4:-2])].lower()
    return paths.repository / "observations" / night / f"ccd_{ccd}" / f"{day}{month}{ccd}{int(run):04d}.fits"


def _parse_log_line(line):
    """Parse one fixed-format observing-log line."""
    run = line[:4]
    if not run.isnumeric():
        return None
    ccd, colon = line[6], line.find(":")
    if colon < 0:
        return None
    overscan_tokens = line[colon + 70:].split()
    overscan = overscan_tokens[0] if overscan_tokens else ""
    return {
        "run": run, "ccd": ccd, "object_log": line[8:colon - 2].strip(),
        "utc_log": line[colon - 2:colon + 7].strip(),
        "exptime_log": line[colon + 9:colon + 17].strip(),
        "lc_status_log": line[colon + 35:colon + 37].strip(),
        "thxe_status_log": line[colon + 38:colon + 42].strip(),
        "overscan_log": overscan,
        "comments": line[colon + 71 + len(overscan):].strip(),
        "marked_bad": "crap" in line.lower() or "unknown" in line.lower(),
    }



def parse_observing_log(filename):
    """Parse an observing log and group entries by run and CCD."""
    runs = {}
    with open(filename) as handle:
        for line in handle:
            row = _parse_log_line(line)
            if row:
                runs.setdefault(row["run"], {})[row["ccd"]] = row
    return runs


def classify_object(name):
    """Map an observing-log object name to an exposure type."""
    return {
        "SimLC": "SimLC", "BiasFrame": "Bias", "FlatField-Quartz": "Flat",
        "ARC-ThAr": "FibTh", "SimTh": "SimTh", "SimThLong": "SimTh",
        "DarkFrame": "Dark", "Acquire": "Acquire",
    }.get(name, "Science")


def useful_ccds(kind, exptime, atol=0.01):
    """Return CCDs usable for this exposure type and exposure time."""
    if kind in CALIBRATION_EXPTIMES:
        return tuple(
            ccd for ccd, required in CALIBRATION_EXPTIMES[kind].items()
            if np.isclose(exptime, required, atol=atol, rtol=0)
        )
    return EXPECTED_CCDS.get(kind, ())


def _to_float(value, default=np.nan):
    """Convert to float, returning a default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_exposure_metadata(files):
    """Read metadata from the first available CCD FITS header."""
    header, header_ccd = None, ""
    for ccd in ("3", "2", "1"):
        filename = files.get(ccd)
        if filename is not None and Path(filename).exists():
            header, header_ccd = fits.getheader(filename, 0), ccd
            break
    if header is None:
        return {
            "header_ccd": "", "run_fits": -1, "object_fits": "", "gaia_id": "",
            "date_obs": "", "mjd_obs": np.nan, "mjd_mid": np.nan, "exptime": np.nan,
            "ut_start": "", "ut_end": "", "ra": np.nan, "dec": np.nan, "airmass": np.nan,
            "lc_requested": False, "lc_ut": "", "lc_exp": np.nan,
        }

    mjd_obs, exptime = _to_float(header.get("MJD-OBS")), _to_float(header.get("EXPTIME"))
    mjd_mid = mjd_obs + 0.5 * exptime / 86400 if np.isfinite(mjd_obs) and np.isfinite(exptime) else np.nan
    lc_ut = str(header.get("LCUT", "")).strip()
    return {
        "header_ccd": header_ccd, "run_fits": int(_to_float(header.get("RUN"), -1)),
        "object_fits": str(header.get("OBJECT", "")).strip(), "gaia_id": str(header.get("GAIAID", "")).strip(),
        "date_obs": str(header.get("DATE-OBS", "")).strip(), "mjd_obs": mjd_obs, "mjd_mid": mjd_mid,
        "exptime": exptime, "ut_start": str(header.get("UTSTART", "")).strip(),
        "ut_end": str(header.get("UTEND", "")).strip(), "ra": _to_float(header.get("MEANRA")),
        "dec": _to_float(header.get("MEANDEC")), "airmass": _to_float(header.get("AIRMASS")),
        "lc_requested": bool(lc_ut), "lc_ut": lc_ut, "lc_exp": _to_float(header.get("LCEXP")),
    }


def build_observation_table(log_runs, config, paths):
    """Combine log and FITS information into the observation table."""
    rows = []
    for run in sorted(log_runs, key=int):
        ccd_log = log_runs[run]
        log = ccd_log["3"] if "3" in ccd_log else next(iter(ccd_log.values()))
        kind = classify_object(log["object_log"])
        files = {ccd: raw_fits_path(paths, config.night, run, ccd) for ccd in ("1", "2", "3")}
        has = {ccd: files[ccd].exists() for ccd in files}
        meta = read_exposure_metadata(files)

        # Calibration exposure time determines which CCDs contain useful data.
        useful = useful_ccds(kind, meta["exptime"])
        use_ccd = {ccd: ccd in useful and has[ccd] for ccd in files}
        missing = [ccd for ccd in EXPECTED_CCDS.get(kind, ()) if not has[ccd]]

        issues = []
        if log["marked_bad"]:
            issues.append("Marked bad in observing log")
        if missing:
            issues.append("Missing CCD" + ", CCD".join(missing))
        if kind in CALIBRATION_EXPTIMES and not useful:
            issues.append(f"Unexpected exposure time {meta['exptime']:.2f} s")

        rows.append({
            "run": run, "type": kind, "object": log["object_log"], "exptime": meta["exptime"],
            "mjd_obs": meta["mjd_obs"], "mjd_mid": meta["mjd_mid"], "ra": meta["ra"],
            "dec": meta["dec"], "airmass": meta["airmass"],
            **{f"has_ccd{c}": has[c] for c in files}, **{f"use_ccd{c}": use_ccd[c] for c in files},
            **{f"file_ccd{c}": str(files[c]) for c in files}, "files_complete": not missing,
            "use": not log["marked_bad"] and kind != "Acquire", "lc_requested": meta["lc_requested"],
            "lc_ut": meta["lc_ut"], "lc_exp": meta["lc_exp"], "comments": log["comments"],
            "issue": "; ".join(issues),
        })
    return Table(rows=rows)


def assign_calibration_blocks(table, gap_minutes=30.0):
    """Group consecutive calibration exposures into calibration blocks."""
    blocks = np.full(len(table), "", dtype="U16")
    number, previous_is_cal, previous_mjd = 0, False, None

    for i in np.argsort(table["mjd_mid"]):
        row = table[i]
        is_cal = row["type"] in CALIBRATION_TYPES and row["use"]
        if not is_cal:
            previous_is_cal, previous_mjd = False, None
            continue

        new = not previous_is_cal
        if previous_mjd is not None and np.isfinite(row["mjd_mid"]):
            new |= (row["mjd_mid"] - previous_mjd) * 1440 > gap_minutes
        if new:
            number += 1
        
        blocks[i] = f"cal{number:03d}"
        previous_is_cal, previous_mjd = True, row["mjd_mid"]
    
    table["calibration_block"] = blocks
    return table


def write_reduction_input(table, config, paths):
    """Write the human-readable reduction input summary."""

    if "calibration_block" not in table.colnames:
        table = table.copy()
        assign_calibration_blocks(table)

    with open(paths.reduction_input, "w") as handle:
        handle.write(f"VeloceReduction input for night {config.night}\n{'=' * 100}\n\n")
        handle.write(f"{'Run':<6}{'Type':<10}{'Object':<22}{'Exp[s]':>8}{'MJD-mid':>15}  {'CCD1':>4}{'CCD2':>5}{'CCD3':>5}  {'LC':>4}  {'Block':<8}{'Use':>5}\n")
        handle.write("-" * 100 + "\n")
        for row in table:
            exp = f"{row['exptime']:.1f}" if np.isfinite(row["exptime"]) else ""
            mjd = f"{row['mjd_mid']:.6f}" if np.isfinite(row["mjd_mid"]) else ""
            ccd = ["Y" if row[f"use_ccd{i}"] else "-" for i in (1, 2, 3)]
            handle.write(
                f"{row['run']:<6}{row['type']:<10}{row['object'][:21]:<22}{exp:>8}{mjd:>15}  "
                f"{ccd[0]:>4}{ccd[1]:>5}{ccd[2]:>5}  {'Y' if row['lc_requested'] else '-':>4}  "
                f"{row['calibration_block']:<8}{'Y' if row['use'] else 'N':>5}\n"
            )
            if row["issue"]:
                handle.write(f"      WARNING: {row['issue']}\n")
            if row["comments"]:
                handle.write(f"      Comment: {row['comments']}\n")


def identify_observations(config, paths):
    """Identify observations for a night and write the reduction summary."""

    logs = sorted(paths.observations.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No observing log found in {paths.observations}")
    if len(logs) > 1:
        logger.warning("Found %d observing logs; using %s", len(logs), logs[0].name)

    shutil.copy2(logs[0], paths.root / logs[0].name)
    table = assign_calibration_blocks(build_observation_table(parse_observing_log(logs[0]), config, paths))
    write_reduction_input(table, config, paths)

    logger.info("Identified %d observing runs", len(table))
    for kind in sorted(set(np.asarray(table["type"]).astype(str))):
        use = (np.asarray(table["type"]).astype(str) == kind) & np.asarray(table["use"], bool)
        logger.info("  %-8s: %d usable runs", kind, int(use.sum()))

    issues = np.count_nonzero(np.asarray(table["issue"]).astype(str) != "")
    if issues:
        logger.warning("Observation table contains %d runs with warnings; see %s", issues, paths.reduction_input)
    return table


def select(table, kind, ccd=None):
    """Select usable observations of a given type and optionally CCD."""
    if len(table) == 0: return table
    mask = (table["type"] == kind) & table["use"]
    if ccd is not None:
        mask &= table[f"use_ccd{ccd}"]
    return table[mask]
