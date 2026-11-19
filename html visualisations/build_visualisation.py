"""
Build the AS/NZS 4777.2 response explorer (one self-contained HTML file).

    python "html visualisations/build_visualisation.py"

Writes, next to this script:
    as4777_explorer_internal.html    every source in SOURCES (contains real site data: internal use only)
    as4777_explorer_synthetic.html   synthetic sources only (safe to share)

Nothing is re-implemented here: data come from standard_plots/lib/sources.py,
units and metadata lines from standard_plots/lib/a_site_day.py, set-points from
bms_sa_review/shared/ciccada_config.py and colours from standard_plots/lib/palette.py.
The page re-computes the curves in JavaScript; a table of answers from
bms_sa_review/shared/as4777_curves.py is embedded so the page checks itself on load.

Edit SOURCES to change what the page offers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (REPO / "standard_plots", REPO, REPO / "bms_sa_review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib import a_site_day as A  # noqa: E402  (standard_plots/lib)
from lib import palette as C  # noqa: E402
from lib import sources as S  # noqa: E402
from shared import as4777_curves as K  # noqa: E402  (bms_sa_review/shared)
from shared.ciccada_config import AS4777  # noqa: E402

# ---------------------------------------------------------------------------
# What the page offers. kind = "local" (site_id, date) | "aws" (site_id, date)
# | "synthetic" (example = "1P" or "3P"). The first entry opens by default.
# ---------------------------------------------------------------------------
SOURCES = [
    # --- single-phase, real
    dict(
        kind="local",
        site_id=1808312180,
        date="2024-12-25",
        note="Volt-VAr and Volt-Watt both conform",
    ),
    dict(
        kind="local",
        site_id=993464143,
        date="2024-12-10",
        note="Volt-VAr and Volt-Watt both conform",
    ),
    dict(
        kind="local",
        site_id=873861233,
        date="2024-12-17",
        note="Volt-VAr conforms, Volt-Watt closest to failing",
    ),
    dict(
        kind="local",
        site_id=1295784779,
        date="2024-12-23",
        note="Volt-VAr fails, Volt-Watt conforms",
    ),
    dict(
        kind="local",
        site_id=2066603468,
        date="2024-12-27",
        note="Volt-VAr fails, Volt-Watt conforms",
    ),
    dict(
        kind="local",
        site_id=465008538,
        date="2025-01-17",
        note="Volt-VAr and Volt-Watt both fail",
    ),
    # --- three-phase, real
    dict(
        kind="local",
        site_id=1327502909,
        date="2025-01-24",
        note="Volt-VAr and Volt-Watt both conform",
    ),
    dict(
        kind="local",
        site_id=741657766,
        date="2025-01-13",
        note="Volt-VAr fails, Volt-Watt conforms",
    ),
    dict(
        kind="local",
        site_id=530210663,
        date="2024-12-29",
        note="Volt-VAr fails, Volt-Watt conforms",
    ),
    dict(
        kind="local",
        site_id=1598853781,
        date="2025-01-25",
        note="Volt-VAr and Volt-Watt both fail",
    ),
    # --- synthetic
    dict(
        kind="synthetic",
        example="1P",
        note="invented data, responds at 75% of the curve",
    ),
    dict(
        kind="synthetic", example="3P", note="invented data, one phase does not respond"
    ),
]

CLOUD_MIN_P_PCT = (
    2.0  # month cloud keeps intervals with P above this (% of rating); drops night
)
TEMPLATE = HERE / "template.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(values, nd=2):
    """Series/array -> list with NaN as None, rounded."""
    out = []
    for x in np.asarray(values, dtype=float):
        out.append(None if not math.isfinite(x) else round(float(x), nd))
    return out


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(pd.Timestamp(v).date())
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)


def _load(spec):
    """(day_df, day_info, month_df, month_label) for one SOURCES entry."""
    kind = spec["kind"]
    if kind == "synthetic":
        day, info = S.load_synthetic(spec["example"])
        month, minfo = S.load_synthetic_month(spec["example"])
        return day, info, month, minfo["date"]
    if kind in ("local", "aws"):
        loader = S.load_local if kind == "local" else S.load_aws
        day, info = loader(spec["site_id"], spec["date"])
        month_label = str(spec["date"])[:7]
        month, _ = S.load_month(spec["site_id"], month_label, source=kind)
        return day, info, month, month_label
    raise ValueError(f"Unknown source kind {kind!r}")


def build_source(spec):
    day_df, info, month_df, month_label = _load(spec)
    rating = float(info["rating_kva"])
    day = A.prepare(day_df, rating, info)  # site totals, % of rating, mean V
    month = A.prepare(month_df, rating, info)

    minutes = (day.t.hour * 60 + day.t.minute).to_numpy()
    keep = (month.p_pct > CLOUD_MIN_P_PCT) & month.v.notna() & month.q_pct.notna()
    synthetic = spec["kind"] == "synthetic"
    site = {
        k: _jsonable(v)
        for k, v in (info.get("site") or {}).items()
        if k != "pv_circuit_ids"
    }
    return {
        "id": f"{spec['kind']}-{spec.get('site_id', spec.get('example'))}-{info['date']}",
        "label": info["label"] + f" · {info['date']}",
        "synthetic": synthetic,
        "note": spec.get("note", ""),
        "rating_kva": rating,
        "phases": len(day.phases),
        "date": info["date"],
        "month": month_label,
        "meta_lines": A.metadata_lines(day),
        "site": site,
        "day": {
            "t": [int(x) for x in minutes],
            "v": _clean(day.v),
            "p": _clean(day.p_pct),
            "q": _clean(day.q_pct),
            "pot": _clean(day.p_pot_pct) if day.p_pot_pct is not None else None,
        },
        "cloud": {
            "v": _clean(month.v[keep]),
            "p": _clean(month.p_pct[keep]),
            "q": _clean(month.q_pct[keep]),
        },
    }


def self_check_table():
    """Answers from as4777_curves.py for S = 100 (so values read as % of rating)."""
    S100, tol = 100.0, AS4777["TOL_FRAC"] * 100.0
    rows = []
    for v in np.arange(200.0, 266.01, 1.5):
        for p in (0, 10, 19.9, 20, 40, 60, 60.1, 70, 80, 80.1, 90, 97, 100, 105):
            q_req = K.vvar_required_q(v, S100)
            floor = K.q_conformance_floor_absorbing(p, S100)
            vmax, vmin = q_req + tol, q_req - tol
            qmax = max(vmax, floor + tol) if vmax < 0 else vmax
            qmin = min(vmin, -floor - tol) if vmin > 0 else vmin
            rows.append(
                [round(float(v), 2), p, q_req, floor, qmin, qmax, K.vw_max_p(v, S100)]
            )
    return {
        "columns": ["v", "p", "q_req", "floor", "q_min_final", "q_max_final", "vw_max"],
        "rows": rows,
    }


def payload(specs):
    return {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "standard": "AS/NZS 4777.2:2020, Australia A",
        "setpoints": {k: AS4777[k] for k in ("VW", "VVAR", "QCAP", "TOL_FRAC")},
        "colours": {
            "quantity": C.QUANTITY,
            "reference": C.REFERENCE,
            "status": C.STATUS,
            "band_alpha": C.BAND_ALPHA,
        },
        "check": self_check_table(),
        "sources": [build_source(s) for s in specs],
    }


def write_page(specs, out: Path, internal: bool):
    data = payload(specs)
    data["internal"] = internal
    html = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*__PAYLOAD__*/null"
    if marker not in html:
        raise RuntimeError("template.html has no payload marker")
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    out.write_text(html.replace(marker, blob), encoding="utf-8")
    n_real = sum(not s["synthetic"] for s in data["sources"])
    print(
        f"Wrote {out.name}: {len(data['sources'])} sources ({n_real} real), {out.stat().st_size / 1024:.0f} kB"
    )
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--synthetic-only",
        action="store_true",
        help="skip the internal page (no data access needed)",
    )
    ap.add_argument(
        "--dump-payload",
        type=Path,
        help="also write the internal payload as JSON (debugging)",
    )
    args = ap.parse_args()

    synth = [s for s in SOURCES if s["kind"] == "synthetic"]
    write_page(synth, HERE / "as4777_explorer_synthetic.html", internal=False)
    if not args.synthetic_only:
        data = write_page(
            SOURCES, HERE / "as4777_explorer_internal.html", internal=True
        )
        if args.dump_payload:
            args.dump_payload.write_text(
                json.dumps(data, allow_nan=False), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
