"""Run the complete SAPN November 2022 conformance workflow."""

from functools import partial
from pathlib import Path

from config import (
    GENERATE_METHOD_COMPARISON_PLOTS,
    GENERATE_SITE_PLOTS_DEFAULT,
)
from core.pipeline import run_conformance
from reporting.generate_method_plots import generate_method_plots
from reporting.outputs import write_outputs
from sapn2022_workflow.sapn_paths import CONFORMANCE_OUTPUT_DIR
from sapn2022_workflow.workflow import load_sapn2022_inputs, prepare_sapn2022_site


def main(output_dir=CONFORMANCE_OUTPUT_DIR, generate_site_plots=None):
    output_dir = Path(output_dir)
    if generate_site_plots is None:
        generate_site_plots = GENERATE_SITE_PLOTS_DEFAULT

    inputs = load_sapn2022_inputs()
    # Bind the shared SAPN data once. The pipeline supplies each candidate site ID
    # and receives either a prepared site or a recorded skip reason.
    prepare_site = partial(prepare_sapn2022_site, inputs=inputs)
    results = run_conformance(
        candidate_site_ids=inputs["candidate_site_ids"],
        prepare_site=prepare_site,
        generate_site_plots=generate_site_plots,
        site_plot_dir=output_dir / "overall_site_plots",
    )
    write_outputs(results, output_dir)
    if GENERATE_METHOD_COMPARISON_PLOTS:
        generate_method_plots(output_dir=output_dir)

    print("Saved outputs to", output_dir)
    skipped = results["skipped_sites"]
    print("Skipped (site metadata rows != 1):", len(skipped["not_single_inverter"]))
    print("Skipped (>3 PV circuits):", len(skipped["more_than_3_pv_circuits"]))
    print("Skipped (no pv_site_net circuits):", len(skipped["no_pv_site_net"]))
    print("Skipped (no day data):", len(skipped["no_day_data"]))
    print("Skipped (no eligible days):", len(skipped["no_eligible_days"]))
    return results


if __name__ == "__main__":
    main()
