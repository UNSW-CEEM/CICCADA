"""Dataset-neutral orchestration of a complete conformance run."""

import polars as pl
from config import PHASE_B_METHODS, PRIMARY_PHASE_B_METHOD
from core.phase_a import run_phase_a_for_site
from core.phase_b import run_phase_b_for_site
from reporting.plotting import plot_site_compliance_day


def _concat_or_empty(frames):
    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()


def run_conformance(
    candidate_site_ids,
    prepare_site,
    *,
    methods=PHASE_B_METHODS,
    primary_method=PRIMARY_PHASE_B_METHOD,
    include_by_method_outputs=False,
    generate_site_plots=False,
    plot_no_eligible_timestamp_days=False,
    site_plot_dir=None,
):
    """Run Phase A once and Phase B across one dataset.

    ``candidate_site_ids`` contains the sites represented in the cleaned data.
    ``prepare_site`` is called with one site ID at a time and must return either
    its prepared day data or a skip reason. By default only ``primary_method``
    is run; set ``include_by_method_outputs`` to run and retain every method.
    """
    threshold_rows = []
    threshold_rows_by_method = []
    phase_a_records = []
    bracket_rows = []
    phase_b_summary_rows = []
    phase_b_detail_rows = []
    phase_b_summary_rows_by_method = []
    phase_b_detail_rows_by_method = []
    excluded_day_rows = []
    skipped_sites = {
        "not_single_inverter": [],
        "more_than_3_pv_circuits": [],
        "no_pv_site_net": [],
        "no_day_data": [],
        "no_eligible_days": [],
        "missing_rated_capacity": [],
    }

    for index, site_number in enumerate(candidate_site_ids, start=1):
        prepared_site = prepare_site(site_number)
        excluded_day_rows.extend(prepared_site.get("excluded_day_rows", []))
        skip_reason = prepared_site.get("skip_reason")
        if skip_reason is not None:
            skipped_sites[skip_reason].append(site_number)
            continue

        day_behaviours = prepared_site["day_behaviours"]
        p_rated = prepared_site["p_rated"]
        phase_a = run_phase_a_for_site(site_number, day_behaviours, p_rated)
        if not phase_a["records"].is_empty():
            phase_a_records.append(phase_a["records"])
        if not phase_a["brackets"].is_empty():
            bracket_rows.append(phase_a["brackets"])

        primary_phase_b = None
        methods_to_run = methods if include_by_method_outputs else (primary_method,)
        for method_key in methods_to_run:
            method_phase_b = run_phase_b_for_site(
                site_number,
                day_behaviours,
                p_rated,
                raw_thresholds=phase_a["raw_thresholds"],
                confidence_info=phase_a["confidence_info"],
                phase_b_method=method_key,
            )
            if include_by_method_outputs:
                threshold_rows_by_method.append(
                    method_phase_b["threshold_row"].with_columns(
                        pl.lit(method_key).alias("method_key")
                    )
                )
                phase_b_summary_rows_by_method.append(
                    method_phase_b["summary_row"].with_columns(
                        pl.lit(method_key).alias("method_key")
                    )
                )
                if not method_phase_b["detail"].is_empty():
                    phase_b_detail_rows_by_method.append(
                        method_phase_b["detail"].with_columns(
                            pl.lit(method_key).alias("method_key")
                        )
                    )
            if method_key == primary_method:
                primary_phase_b = method_phase_b

        if primary_phase_b is None:
            raise ValueError(f"Primary method {primary_method!r} was not run")

        primary_thresholds = primary_phase_b["threshold_row"].to_dicts()[0]
        threshold_rows.append(primary_phase_b["threshold_row"])
        phase_b_summary_rows.append(primary_phase_b["summary_row"])
        if not primary_phase_b["detail"].is_empty():
            phase_b_detail_rows.append(primary_phase_b["detail"])

        summary = primary_phase_b["summary_row"].to_dicts()[0]
        if (
            generate_site_plots
            and site_plot_dir is not None
            and summary["overall_pass"] is not None
        ):
            plot_folder = (
                "compliant" if summary["overall_pass"] is True else "non_compliant"
            )
            for day_info in day_behaviours:
                day_plot = day_info["behaviour"].phase_b_day(
                    p_rated,
                    los_threshold=summary["los_threshold_used"],
                    ov1_work_threshold=primary_thresholds["ov1_work_site"],
                )
                plot_site_compliance_day(
                    day_plot["frame"],
                    site_number,
                    day_info["day"],
                    p_rated=p_rated,
                    lso_threshold=summary["los_threshold_used"],
                    ov1_threshold=primary_thresholds["ov1_test_site"],
                    overall_pass=summary["overall_pass"],
                    day_summary=day_plot["summary"],
                    plot_no_eligible_timestamp_days=plot_no_eligible_timestamp_days,
                    save_path=(
                        site_plot_dir
                        / plot_folder
                        / f"Site_{site_number}_Day_{day_info['day']}_{plot_folder}.png"
                    ),
                )

        print(
            f"[{index}/{len(candidate_site_ids)}] site {site_number} "
            f"LOS={summary['los_compliance_pct']} "
            f"OV1={summary['ov1_compliance_pct']} PASS={summary['overall_pass']}"
        )

    return {
        "site_thresholds": _concat_or_empty(threshold_rows),
        "site_thresholds_by_method": _concat_or_empty(threshold_rows_by_method),
        "phase_a_trip_attribution": _concat_or_empty(phase_a_records),
        "phase_a_brackets": _concat_or_empty(bracket_rows),
        "phase_b_site_summary": _concat_or_empty(phase_b_summary_rows),
        "phase_b_timestamp_detail": _concat_or_empty(phase_b_detail_rows),
        "phase_b_site_summary_by_method": _concat_or_empty(
            phase_b_summary_rows_by_method
        ),
        "phase_b_timestamp_detail_by_method": _concat_or_empty(
            phase_b_detail_rows_by_method
        ),
        "excluded_day_rows": excluded_day_rows,
        "skipped_sites": skipped_sites,
    }
