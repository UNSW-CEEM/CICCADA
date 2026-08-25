"""
Athena access, credential diagnosis and scan accounting.
========================================================

Phase 0. The one module that knows how to reach AWS. Everything else asks it.

It does not replace `bms_sa_review.shared.aws_config` -- it wraps it. Same SSO
profile (`ciccada`), same region, same S3 staging bucket, same `aq()`
underneath. Three things are added:

1. **Specific credential failure.** `aws_config` builds a boto3 session at
   import time, so an expired token surfaces as an opaque error from whichever
   cell happened to import it first. `credential_status()` names the failure and
   prints the exact command that fixes it.

2. **A partition guard.** Athena bills by data scanned. A `SELECT` against `ts`
   without a partition predicate is a real cost event, and it is the easiest
   mistake to make. `check_partition_filters()` reads the SQL and refuses.

3. **Scan accounting.** Every query run through `aq()` records what it actually
   scanned, so a notebook can print its own bill instead of estimating one.
   Degrades gracefully: if the scan figure cannot be recovered the query still
   runs and the row is marked unknown.

Typical use
-----------
    from bms_sa_review.ami_data_analysis.lib import ami_athena as A

    A.require_credentials()
    df = A.aq("SELECT * FROM circuits LIMIT 5", database=A.C.SA)
    A.scan_report()
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from bms_sa_review.ami_data_analysis.config import ami_config as C

__all__ = [
    "get_aq", "get_session", "get_aws_config",
    "credential_status", "require_credentials",
    "aq", "describe", "preview_sql",
    "check_partition_filters", "assert_partition_filters",
    "scan_report", "reset_scan_log", "SCAN_LOG",
    "fmt_bytes", "estimate_cost", "billed_bytes",
]

C = C  # re-exported for notebooks: `A.C.SAI`


# ═══════════════════════════════════════════════════════════════════════════
# LAZY ACCESS TO THE EXISTING PLUMBING
# ═══════════════════════════════════════════════════════════════════════════

def get_aws_config():
    """
    Import `shared.aws_config` lazily and turn its failures into readable ones.

    Deliberately lazy: `aws_config` constructs `boto3.Session(profile_name=...)`
    at module scope, so importing it eagerly would make every module in this
    package -- including the local-only Phase 5 and 6 ones -- fail on a missing
    profile or an expired token.
    """
    C.bootstrap_sys_path()
    try:
        from bms_sa_review.shared import aws_config
    except Exception as exc:
        raise RuntimeError(
            "Could not import bms_sa_review.shared.aws_config.\n"
            "  1. aws sso login --profile ciccada\n"
            "  2. check boto3, awswrangler and pandas are installed in this kernel\n"
            f"  original error: {type(exc).__name__}: {exc}"
        ) from exc
    return aws_config


def get_aq():
    """The project's Athena query helper, unwrapped. Prefer `aq()` below."""
    return get_aws_config().aq


def get_session():
    """The project's boto3 session."""
    return get_aws_config().session


# ═══════════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════

#: Exception class names that mean "your SSO session needs refreshing", mapped
#: to the remedy. Matched on class name rather than by importing botocore's
#: exception classes, which move between versions.
_CREDENTIAL_ERRORS = {
    "UnauthorizedSSOTokenError": "SSO token is missing or expired",
    "SSOTokenLoadError": "SSO token could not be loaded",
    "TokenRetrievalError": "SSO token could not be retrieved",
    "ExpiredTokenException": "temporary credentials have expired",
    "CredentialRetrievalError": "credentials could not be retrieved",
    "NoCredentialsError": "no credentials found at all",
    "ProfileNotFound": "the named AWS profile does not exist",
    "UnrecognizedClientException": "credentials were rejected by AWS",
}

_SSO_REMEDY = (
    "Run this in a terminal, then RESTART THE KERNEL:\n"
    "    aws sso login --profile ciccada\n"
    "The kernel caches the expired token, so re-running the cell alone will not help."
)

_PROFILE_REMEDY = (
    "Run `aws configure list-profiles`. If the profile is not called 'ciccada',\n"
    "either rename it or set AWS_PROFILE before starting the kernel."
)


def _classify(exc: BaseException) -> tuple[str, str]:
    """(human reason, remedy) for a credential exception. Pure."""
    for cls in type(exc).__mro__:
        if cls.__name__ in _CREDENTIAL_ERRORS:
            reason = _CREDENTIAL_ERRORS[cls.__name__]
            remedy = _PROFILE_REMEDY if cls.__name__ == "ProfileNotFound" else _SSO_REMEDY
            return reason, remedy
    text = str(exc).lower()
    if "token" in text and ("expire" in text or "invalid" in text):
        return "SSO token appears to have expired", _SSO_REMEDY
    return f"{type(exc).__name__}", _SSO_REMEDY


def credential_status() -> dict:
    """
    Whether the SSO session works, and if not, exactly why and what to do.

    Never raises. Returns a dict with `ok`, `profile`, `region`, `account`,
    `arn`, `reason`, `remedy`.
    """
    out = {
        "ok": False, "profile": None, "region": None,
        "account": None, "arn": None, "reason": "", "remedy": "",
    }
    try:
        cfg = get_aws_config()
    except RuntimeError as exc:
        out["reason"], out["remedy"] = _classify(exc.__cause__ or exc)
        out["reason"] = f"import failed: {out['reason']}"
        return out

    out["profile"] = getattr(cfg, "PROFILE", None)
    out["region"] = getattr(cfg, "REGION", None)

    try:
        ident = cfg.session.client("sts").get_caller_identity()
    except Exception as exc:
        out["reason"], out["remedy"] = _classify(exc)
        return out

    out.update(ok=True, account=ident.get("Account"), arn=ident.get("Arn"),
               reason="valid")
    return out


def require_credentials() -> dict:
    """`credential_status()`, but raises with the remedy if the session is dead."""
    status = credential_status()
    if not status["ok"]:
        raise RuntimeError(
            f"AWS credentials are not usable: {status['reason']}\n"
            f"  profile: {status['profile']}   region: {status['region']}\n\n"
            f"{status['remedy']}"
        )
    return status


# ═══════════════════════════════════════════════════════════════════════════
# COST ARITHMETIC  (pure)
# ═══════════════════════════════════════════════════════════════════════════

def fmt_bytes(n: float | int | None) -> str:
    """Human-readable byte count. `None` renders as '?'."""
    if n is None:
        return "?"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 2
            return f"{sign}{n:,.{precision}f} {unit}"
        n /= 1024
    return f"{sign}{n:,.2f} TB"  # pragma: no cover - unreachable


def billed_bytes(scanned: float | int | None) -> float | None:
    """
    What Athena bills for a query that scanned `scanned` bytes.

    Athena rounds each query up to a 10 MB minimum. A hundred "free" metadata
    queries therefore cost about a cent, not nothing.
    """
    if scanned is None:
        return None
    return max(float(scanned), float(C.ATHENA_MIN_SCAN_BYTES))


def estimate_cost(scanned: float | int | None, apply_minimum: bool = True) -> float | None:
    """Money for a scan, in `C.ATHENA_PRICE_CURRENCY`."""
    if scanned is None:
        return None
    payable = billed_bytes(scanned) if apply_minimum else float(scanned)
    return payable / 1024**4 * C.ATHENA_PRICE_PER_TB


# ═══════════════════════════════════════════════════════════════════════════
# PARTITION GUARD  (pure)
# ═══════════════════════════════════════════════════════════════════════════

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_TABLE_REF = re.compile(r"""\b(?:FROM|JOIN)\s+("?[A-Za-z_][\w.$]*"?)""", re.IGNORECASE)
# NOTE the alternation: a trailing \b after `>=` never matches, because `=` and
# the space that follows it are both non-word characters. That bug silently
# turned every `WHERE t_stamp >= TIMESTAMP '...'` query into a guard failure.
# `\b` belongs on the word `between` alone.
_TIME_BOUND = re.compile(
    r"\b(?:t_stamp|time)\s*(?:>=|<=|>|<|\bbetween\b)", re.IGNORECASE
)


def _strip_noise(sql: str) -> str:
    """SQL with comments and string literals removed. Pure."""
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    return _STRING_LITERAL.sub("''", sql)


def _referenced_tables(sql: str) -> list[str]:
    """
    Bare table names appearing after FROM or JOIN. Pure.

    Database qualifiers are dropped and quotes stripped, so
    `solar_analytics_iceberg."ts"` and `ts` both come back as `ts`.
    Sub-selects (`FROM (SELECT ...`) do not match and are not returned.
    """
    names = []
    for raw in _TABLE_REF.findall(_strip_noise(sql)):
        name = raw.strip('"').split(".")[-1].strip('"')
        if name:
            names.append(name)
    return names


def check_partition_filters(sql: str) -> list[str]:
    """
    Problems with a query's partition predicates, as a list of messages. Pure.

    Empty list means the query is safe to run. A query is flagged when it reads a
    table in `ami_config.BIG_TABLES` without mentioning any partition column
    (`year`, `month`, `is_pv`) or bounding `t_stamp` / `time`.

    Metadata tables (`"ts$partitions"`, `"ts$files"`, ...) are exempt: they read
    Iceberg manifests, not data, and scan effectively nothing.

    Deliberately syntactic and deliberately conservative. It cannot tell a
    genuine partition predicate from the word `year` appearing in a column alias,
    so it will pass some queries it should not. It exists to catch the
    `SELECT ... FROM ts` with no WHERE clause at all, which is the mistake that
    actually costs money.
    """
    body = _strip_noise(sql)
    referenced = _referenced_tables(sql)

    big = sorted({
        name for name in referenced
        if "$" not in name and name.lower() in C.BIG_TABLES
    })
    if not big:
        return []

    has_partition = any(
        re.search(rf"\b{col}\b", body, re.IGNORECASE) for col in C.TS_PARTITION_COLUMNS
    )
    has_time_bound = bool(_TIME_BOUND.search(body))
    if has_partition or has_time_bound:
        return []

    return [
        f"query reads {', '.join(big)} with no partition predicate; "
        f"expected a filter on one of {', '.join(C.TS_PARTITION_COLUMNS)} "
        f"or a bound on t_stamp"
    ]


def assert_partition_filters(sql: str) -> None:
    """Raise if `check_partition_filters` finds anything."""
    problems = check_partition_filters(sql)
    if problems:
        raise ValueError(
            "Refusing to run an unpartitioned query against a large table.\n  "
            + "\n  ".join(problems)
            + "\n\nAdd a partition predicate, or pass allow_full_scan=True if you "
              "have decided the cost is acceptable."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SCAN ACCOUNTING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScanRecord:
    label: str
    database: str
    n_rows: int
    seconds: float
    scanned_bytes: float | None
    source: str            # how the scan figure was obtained
    sql: str = field(repr=False, default="")


SCAN_LOG: list[ScanRecord] = []


def reset_scan_log() -> None:
    """Clear the accumulated scan records. Call at the top of a notebook."""
    SCAN_LOG.clear()


def _scan_bytes_from_frame(frame) -> tuple[float | None, str]:
    """
    Read the scan figure off the DataFrame awswrangler returned. Pure.

    `wr.athena.read_sql_query(..., ctas_approach=False)` attaches the raw
    `GetQueryExecution` payload as `df.query_metadata`. That attribute is not a
    documented contract, so this never raises -- it returns `(None, "unavailable")`
    and the caller falls back to the API.
    """
    meta = getattr(frame, "query_metadata", None)
    if not isinstance(meta, dict):
        return None, "unavailable"
    payload = meta.get("QueryExecution", meta)
    if not isinstance(payload, dict):
        return None, "unavailable"
    stats = payload.get("Statistics")
    if not isinstance(stats, dict):
        return None, "unavailable"
    value = stats.get("DataScannedInBytes")
    if value is None:
        return None, "unavailable"
    return float(value), "query_metadata"


def _scan_bytes_from_api(sql: str, max_lookback: int = 25) -> tuple[float | None, str]:
    """
    Recover the scan figure by asking Athena about recent query executions.

    Fallback for when `query_metadata` is absent. Matches on the exact query
    text and takes the most recent match, so a concurrent query from another
    session cannot be mistaken for this one. Free (Athena API calls are not
    billed by scan). Never raises.
    """
    try:
        client = get_session().client("athena")
        listed = client.list_query_executions(MaxResults=max_lookback)
        ids = listed.get("QueryExecutionIds", [])
        if not ids:
            return None, "unavailable"
        detail = client.batch_get_query_execution(QueryExecutionIds=ids)
        wanted = " ".join(sql.split())
        for execution in detail.get("QueryExecutions", []):
            if " ".join(str(execution.get("Query", "")).split()) == wanted:
                value = execution.get("Statistics", {}).get("DataScannedInBytes")
                if value is not None:
                    return float(value), "athena_api"
        return None, "unavailable"
    except Exception:
        return None, "unavailable"


def aq(
    sql: str,
    database: str | None = None,
    *,
    label: str | None = None,
    allow_full_scan: bool = False,
    meter: bool = True,
) -> pd.DataFrame:
    """
    Run an Athena query, guarded and metered.

    Parameters
    ----------
    database        : defaults to `solar_analytics_iceberg` (the primary catalogue).
    label           : short name for the scan report. Defaults to the first line.
    allow_full_scan : bypass the partition guard. Prints a loud warning; use it
                      only when you have decided the cost is acceptable.
    meter           : record the scan in `SCAN_LOG`. Set False for a query whose
                      cost you do not want counted (there is rarely a reason).
    """
    database = database or C.SAI
    if allow_full_scan:
        problems = check_partition_filters(sql)
        if problems:
            print("WARNING: partition guard bypassed --", problems[0])
    else:
        assert_partition_filters(sql)

    runner = get_aq()
    start = time.perf_counter()
    frame = runner(sql, database=database)
    elapsed = time.perf_counter() - start

    if meter:
        scanned, source = _scan_bytes_from_frame(frame)
        if scanned is None:
            scanned, source = _scan_bytes_from_api(sql)
        SCAN_LOG.append(
            ScanRecord(
                label=label or " ".join(sql.strip().split())[:60],
                database=database,
                n_rows=len(frame),
                seconds=round(elapsed, 2),
                scanned_bytes=scanned,
                source=source,
                sql=sql,
            )
        )
    return frame


def scan_report(verbose: bool = True) -> pd.DataFrame:
    """
    Every query this session ran, what it scanned, and what it cost.

    The `source` column matters: `unavailable` means the scan figure could not be
    recovered, NOT that the query was free. Totals are therefore lower bounds.
    """
    if not SCAN_LOG:
        if verbose:
            print("No metered queries yet.")
        return pd.DataFrame(
            columns=["label", "database", "n_rows", "seconds",
                     "scanned", "scanned_bytes", "cost", "source"]
        )

    rows = []
    for record in SCAN_LOG:
        cost = estimate_cost(record.scanned_bytes)
        rows.append({
            "label": record.label,
            "database": record.database,
            "n_rows": record.n_rows,
            "seconds": record.seconds,
            "scanned": fmt_bytes(record.scanned_bytes),
            "scanned_bytes": record.scanned_bytes,
            "cost": None if cost is None else round(cost, 4),
            "source": record.source,
        })
    frame = pd.DataFrame(rows)

    if verbose:
        known = frame.scanned_bytes.dropna()
        n_unknown = int(frame.scanned_bytes.isna().sum())
        total_bytes = float(known.sum())
        total_cost = sum(
            estimate_cost(value) or 0.0 for value in known
        )
        print(f"{len(frame)} quer{'y' if len(frame) == 1 else 'ies'}, "
              f"{fmt_bytes(total_bytes)} scanned, "
              f"~{C.ATHENA_PRICE_CURRENCY} {total_cost:.4f} "
              f"(billed at a {fmt_bytes(C.ATHENA_MIN_SCAN_BYTES)} minimum per query)")
        if n_unknown:
            print(f"  {n_unknown} quer{'y' if n_unknown == 1 else 'ies'} could not "
                  "report a scan figure -- the total above is a LOWER BOUND.")
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# SMALL CONVENIENCES
# ═══════════════════════════════════════════════════════════════════════════

def describe(table: str, database: str | None = None) -> pd.DataFrame:
    """A table's columns and types. Metadata only -- scans no data."""
    return aq(f"DESCRIBE {table}", database=database or C.SAI,
              label=f"DESCRIBE {table}")


def preview_sql(sql: str) -> None:
    """Print SQL with line numbers, without running it."""
    lines = sql.strip().splitlines()
    width = len(str(len(lines)))
    for number, line in enumerate(lines, start=1):
        print(f"{number:>{width}} | {line}")
