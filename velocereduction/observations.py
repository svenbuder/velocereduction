"""Identify and classify Veloce observations for reduction."""

from calendar import month_abbr
from pathlib import Path
import logging
import shutil
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .constants import (
    CALIBRATION_EXPTIMES, CALIBRATION_TYPES, EXPECTED_CCDS, REFERENCE_NIGHT,
)

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
    simul = line[55:70]
    return {
        "run": run, "ccd": ccd, "object_log": line[8:colon - 2].strip(),
        "utc_log": line[colon - 2:colon + 7].strip(),
        "exptime_log": line[colon + 9:colon + 17].strip(),
        "lc_status_log": "LC" if "LC" in simul.split() else "",
        "thxe_status_log": "ThXe" if "ThXe" in simul.split() else "",
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
            "use": not log["marked_bad"] and kind != "Acquire",
            # Retain the recorded LC flag for diagnostics only. It is not used
            # for block identification because the status is not trustworthy.
            "lc_requested": meta["lc_requested"] or log["lc_status_log"] == "LC",
            "lc_ut": meta["lc_ut"], "lc_exp": meta["lc_exp"], "comments": log["comments"],
            "issue": "; ".join(issues),
        })
    return Table(rows=rows)


def _append_issue(table, index, message):
    current = str(table["issue"][index]).strip()
    table["issue"][index] = f"{current}; {message}" if current else message


def _find_setup_start(table):
    """Find the first credible standard setup sequence before Science begins."""
    kinds = np.asarray(table["type"]).astype(str)
    use = np.asarray(table["use"], bool)
    science = np.where((kinds == "Science") & use)[0]
    stop = int(science[0]) if len(science) else len(table)

    # Normal setup: Flats followed by FibTh and/or SimTh.
    for i in range(stop):
        if kinds[i] != "Flat" or not use[i]:
            continue
        later = kinds[i + 1:stop][use[i + 1:stop]]
        if np.any(later == "FibTh") or np.any(later == "SimTh"):
            return i

    # Allow nights where Flats were skipped: FibTh followed by SimTh.
    for i in range(stop):
        if kinds[i] != "FibTh" or not use[i]:
            continue
        later = kinds[i + 1:stop][use[i + 1:stop]]
        if np.any(later == "SimTh"):
            return i
    return None


def _mark_pre_setup_tests(table):
    """Exclude unlabelled setup tests preceding a credible start calibration sequence."""
    table["issue"] = np.asarray(table["issue"]).astype("U256")
    calibration_use = np.asarray(table["use"], bool).copy()
    start = _find_setup_start(table)
    if start is not None:
        kinds = np.asarray(table["type"]).astype(str)
        for i in range(start):
            if kinds[i] in ("Flat", "FibTh", "SimTh", "SimLC"):
                calibration_use[i] = False
                _append_issue(table, i, "Ignored pre-setup calibration/test")
    table["calibration_use"] = calibration_use


def _simlc_candidate(table, index):
    """Return whether the SimLC region should be checked for this exposure.

    The recorded LC status is deliberately ignored. Dedicated SimLC exposures
    are always checked. Every usable Science exposure is checked individually,
    because the comb may have been illuminated even when the status flag did
    not trigger. Every 15-s SimTh exposure is also checked; the image itself
    decides whether comb light was actually present.
    """
    row = table[index]
    if not bool(row["calibration_use"]):
        return False
    kind = str(row["type"])
    if kind in ("SimLC", "Science"):
        return True
    return (
        kind == "SimTh"
        and np.isfinite(row["exptime"])
        and np.isclose(float(row["exptime"]), 15.0, atol=0.01, rtol=0)
    )


def _block_member(table, index, kind):
    if kind == "SimLC":
        return _simlc_candidate(table, index)
    row = table[index]
    return bool(row["calibration_use"]) and str(row["type"]) == kind


def assign_calibration_blocks(table, gap_minutes=20.0, merged_reference=False):
    """Assign type-specific observing blocks before image-based signal QA.

    Flat/FibTh/SimTh blocks describe the attempted calibration sequence; CCD3
    signal QA later decides which individual exposures actually contribute.
    Dedicated SimLC sequences are grouped regardless of nominal exposure time,
    with CRAP/UNKNOWN rows omitted without splitting the sequence. Science LC
    candidates are singleton blocks tied to their science timestamp. Every 15-s
    SimTh sequence is a separate LC-candidate block. Recorded LC flags are never
    used for these decisions.
    """
    if len(table) == 0:
        return table

    _mark_pre_setup_tests(table)
    kinds = np.asarray(table["type"]).astype(str)
    for kind in CALIBRATION_TYPES:
        labels = np.full(len(table), "", dtype="U24")

        if kind == "SimLC":
            counters = {"SimLC": 0, "SimTh": 0}
            active = False
            previous_context = None
            current_label = ""

            for i, row in enumerate(table):
                raw_member = (
                    kinds[i] in ("SimLC", "Science")
                    or (
                        kinds[i] == "SimTh"
                        and np.isfinite(row["exptime"])
                        and np.isclose(float(row["exptime"]), 15.0, atol=0.01, rtol=0)
                    )
                )
                if not raw_member:
                    active = False
                    previous_context = None
                    current_label = ""
                    continue

                # CRAP/UNKNOWN dedicated calibration rows are omitted but do not
                # break the surrounding attempted sequence.
                if not _block_member(table, i, kind):
                    continue

                context = kinds[i]
                if context == "Science":
                    labels[i] = f"SimLC_run{int(row['run']):04d}"
                    active = False
                    previous_context = None
                    current_label = ""
                    continue

                if not active or context != previous_context:
                    counters[context] += 1
                    current_label = (
                        f"SimLC{counters[context]:03d}"
                        if context == "SimLC"
                        else f"SimLC_SimTh{counters[context]:03d}"
                    )
                labels[i] = current_label
                active = True
                previous_context = context

            table["block_SimLC"] = labels
            continue

        number = 0
        active = False
        previous_mjd = None
        for i, row in enumerate(table):
            raw_member = kinds[i] == kind
            if not raw_member:
                active = False
                previous_mjd = None
                continue

            if not _block_member(table, i, kind):
                if active and np.isfinite(row["mjd_mid"]):
                    previous_mjd = float(row["mjd_mid"])
                continue

            mjd = float(row["mjd_mid"])
            new = not active
            if not merged_reference and active and previous_mjd is not None:
                if np.isfinite(mjd) and np.isfinite(previous_mjd):
                    new |= abs(mjd - previous_mjd) * 1440.0 > gap_minutes
            if new:
                number += 1
            labels[i] = f"{kind}{number:03d}"
            active = True
            previous_mjd = mjd

        table[f"block_{kind}"] = labels

    primary = np.full(len(table), "", dtype="U24")
    for i, kind in enumerate(kinds):
        column = f"block_{kind}"
        if kind in CALIBRATION_TYPES and column in table.colnames:
            primary[i] = table[column][i]
    table["calibration_block"] = primary
    return table

def calibration_blocks(table, kind, ccd=None):
    """Return ``(block_id, rows)`` for one calibration type, optionally one CCD."""
    column = f"block_{kind}"
    if column not in table.colnames:
        raise KeyError(f"{column} is missing; call assign_calibration_blocks first")
    labels = np.asarray(table[column]).astype(str)
    seen, result = set(), []
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        rows = table[labels == label]
        if ccd is not None:
            ccd = str(ccd)
            if ccd not in EXPECTED_CCDS.get(kind, ()):
                rows = rows[:0]
            elif kind == "SimLC":
                rows = rows[np.asarray(rows[f"has_ccd{ccd}"], bool)]
            else:
                rows = rows[np.asarray(rows[f"use_ccd{ccd}"], bool)]
        if len(rows):
            result.append((label, rows))
    return result


def _format_runs(runs):
    values = [int(run) for run in runs]
    if not values:
        return "-"
    groups, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append((start, previous))
        start = previous = value
    groups.append((start, previous))
    return ",".join(
        f"{a:04d}" if a == b else f"{a:04d}-{b:04d}" for a, b in groups
    )


def print_observation_summary(table):
    """Print Science exposures and attempted calibration blocks."""
    print("\nScience exposures")
    science = table[(table["type"] == "Science") & table["use"]]
    if not len(science):
        print("  none")
    for row in science:
        mjd = f"{float(row['mjd_mid']):.6f}" if np.isfinite(row["mjd_mid"]) else "n/a"
        print(f"  run {row['run']}  MJD {mjd}  object {row['object']}")

    print("\nCalibration blocks")
    found = False
    for kind in CALIBRATION_TYPES:
        for block, rows in calibration_blocks(table, kind):
            # Science LC candidates are checked individually from CCD3 later;
            # they are already listed above and would otherwise dominate this summary.
            if kind == "SimLC" and not np.any(np.asarray(rows["type"]).astype(str) == "SimLC"):
                continue
            found = True
            mjd_values = np.asarray(rows["mjd_mid"], float)
            mjd = np.nanmean(mjd_values) if np.any(np.isfinite(mjd_values)) else np.nan
            mean_text = f"{mjd:.6f}" if np.isfinite(mjd) else "n/a"
            ccd_runs = []
            for ccd in EXPECTED_CCDS.get(kind, ()):
                matches = dict(calibration_blocks(table, kind, ccd)).get(block)
                if matches is not None and len(matches):
                    ccd_runs.append(f"CCD{ccd}={_format_runs(matches['run'])}")
            suffix = f"  [{' ; '.join(ccd_runs)}]" if ccd_runs else ""
            context = ""
            if kind == "SimLC" and len(rows):
                types = sorted(set(np.asarray(rows["type"]).astype(str)))
                context = f"  candidate from {'/'.join(types)}"
            print(f"  {block:<10} MJD {mean_text}  runs {_format_runs(rows['run'])}{suffix}{context}")
    if not found:
        print("  none")
    if len(science):
        print("  SimLC region of each Science exposure will be checked individually on CCD3.")
    print("  SimLC region of every 15-s SimTh exposure will also be checked on CCD3.")

def write_reduction_input(table, config, paths):
    """Write the human-readable reduction input summary."""
    if "calibration_block" not in table.colnames:
        table = table.copy()
        assign_calibration_blocks(table, merged_reference=config.night == REFERENCE_NIGHT)

    with open(paths.reduction_input, "w") as handle:
        handle.write(f"VeloceReduction input for night {config.night}\n{'=' * 112}\n\n")
        handle.write(
            f"{'Run':<6}{'Type':<10}{'Object':<22}{'Exp[s]':>8}{'MJD-mid':>15}  "
            f"{'CCD1':>4}{'CCD2':>5}{'CCD3':>5}  {'LCflag':>6}  {'Block':<10}{'LC block':<10}{'Use':>5}\n"
        )
        handle.write("-" * 112 + "\n")
        handle.write("LCflag is diagnostic only; LC block assignment ignores the recorded LC status.\n")
        for row in table:
            exp = f"{row['exptime']:.1f}" if np.isfinite(row["exptime"]) else ""
            mjd = f"{row['mjd_mid']:.6f}" if np.isfinite(row["mjd_mid"]) else ""
            ccd = ["Y" if row[f"use_ccd{i}"] else "-" for i in (1, 2, 3)]
            handle.write(
                f"{row['run']:<6}{row['type']:<10}{row['object'][:21]:<22}{exp:>8}{mjd:>15}  "
                f"{ccd[0]:>4}{ccd[1]:>5}{ccd[2]:>5}  {'Y' if row['lc_requested'] else '-':>6}  "
                f"{row['calibration_block']:<10}{row['block_SimLC']:<10}{'Y' if row['use'] else 'N':>5}\n"
            )
            if row["issue"]:
                handle.write(f"      WARNING: {row['issue']}\n")
            if row["comments"]:
                handle.write(f"      Comment: {row['comments']}\n")


def identify_observations(config, paths):
    """Identify observations for a night, assign blocks, and print/write a summary."""
    logs = sorted(paths.observations.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No observing log found in {paths.observations}")
    if len(logs) > 1:
        logger.warning("Found %d observing logs; using %s", len(logs), logs[0].name)

    shutil.copy2(logs[0], paths.root / logs[0].name)
    table = build_observation_table(parse_observing_log(logs[0]), config, paths)
    assign_calibration_blocks(table, merged_reference=config.night == REFERENCE_NIGHT)
    table.meta["night"] = config.night
    table.meta["calibration_directory"] = str(paths.calibrations)
    write_reduction_input(table, config, paths)
    print_observation_summary(table)

    logger.info("Identified %d observing runs", len(table))
    for kind in sorted(set(np.asarray(table["type"]).astype(str))):
        use = (np.asarray(table["type"]).astype(str) == kind) & np.asarray(table["use"], bool)
        logger.info("  %-8s: %d usable runs", kind, int(use.sum()))

    issues = np.count_nonzero(np.asarray(table["issue"]).astype(str) != "")
    if issues:
        logger.warning("Observation table contains %d runs with warnings; see %s", issues, paths.reduction_input)
    return table


def select(table, kind, ccd=None):
    """Select usable primary observations, including signal QA when available."""
    if len(table) == 0:
        return table
    usable = (
        table["calibration_use"]
        if kind in CALIBRATION_TYPES and "calibration_use" in table.colnames
        else table["use"]
    )
    mask = (table["type"] == kind) & usable
    if kind in CALIBRATION_TYPES and "signal_use" in table.colnames:
        mask &= table["signal_use"]
    if ccd is not None:
        mask &= table[f"use_ccd{ccd}"]
    return table[mask]

