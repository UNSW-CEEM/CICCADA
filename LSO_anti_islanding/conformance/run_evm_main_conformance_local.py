"""Run conformance for a selected preprocessed dataset."""

from functools import partial

from config import (
    GENERATE_SITE_PLOTS_DEFAULT,
    PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
)
from core.pipeline import run_conformance
from core.workflow import prepare_site
from reporting.outputs import write_outputs
from sapn2022_workflow.adapter import SAPN2022_CONFORMANCE_CONFIG
from solar_analytics_workflow.adapter import SOLAR_ANALYTICS_CONFORMANCE_CONFIG

DATASET_CONFORMANCE_CONFIG_REGISTRY = {
    SAPN2022_CONFORMANCE_CONFIG.name: SAPN2022_CONFORMANCE_CONFIG,
    SOLAR_ANALYTICS_CONFORMANCE_CONFIG.name: SOLAR_ANALYTICS_CONFORMANCE_CONFIG,
}


def _get_dataset_conformance_config(dataset):
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string.")
    dataset_key = dataset.strip().lower()
    try:
        return DATASET_CONFORMANCE_CONFIG_REGISTRY[dataset_key]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_CONFORMANCE_CONFIG_REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Supported datasets: {supported}."
        ) from error


def main(dataset, **dataset_options):
    """Run the shared conformance workflow for one named dataset."""
    workflow_config = _get_dataset_conformance_config(dataset)
    inputs = workflow_config.load_inputs(**dataset_options)
    prepare_dataset_site = partial(
        prepare_site,
        inputs=inputs,
        workflow_config=workflow_config,
    )
    results = run_conformance(
        candidate_site_ids=inputs["candidate_site_ids"],
        prepare_site=prepare_dataset_site,
        generate_site_plots=GENERATE_SITE_PLOTS_DEFAULT,
        plot_no_eligible_timestamp_days=PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
        site_plot_dir=workflow_config.output_dir / "overall_site_plots",
    )
    write_outputs(
        results,
        workflow_config.output_dir,
        excluded_day_schema=workflow_config.excluded_day_schema,
    )
    print("Saved outputs to", workflow_config.output_dir)
    skipped = results["skipped_sites"]
    print(
        "Skipped (not single inverter):",
        len(skipped["not_single_inverter"]),
    )
    print("Skipped (>3 PV circuits):", len(skipped["more_than_3_pv_circuits"]))
    print("Skipped (no pv_site_net circuits):", len(skipped["no_pv_site_net"]))
    print("Skipped (no day data):", len(skipped["no_day_data"]))
    print("Skipped (no eligible days):", len(skipped["no_eligible_days"]))
    print(
        "Skipped (missing rated capacity):",
        len(skipped["missing_rated_capacity"]),
    )
    return results


# verify if the following
# from main_run_conformance import main

# results = main(dataset="solar_analytics")
results = main(dataset="sapn2022")
