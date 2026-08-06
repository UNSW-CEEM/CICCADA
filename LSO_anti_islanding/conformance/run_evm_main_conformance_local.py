"""Run conformance for a selected preprocessed dataset."""

from functools import partial

from config import (
    GENERATE_METHOD_COMPARISON_PLOTS,
    GENERATE_SITE_PLOTS_DEFAULT,
    PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
)
from core.pipeline import run_conformance
from core.workflow import prepare_site
from reporting.generate_method_plots import generate_method_plots
from reporting.outputs import write_outputs
from sapn2022_workflow.adapter import SAPN2022_DEFINITION
from solar_analytics_workflow.adapter import SOLAR_ANALYTICS_DEFINITION

# clearly define dataset defintion here

DATASET_DEFINITIONS = {
    SAPN2022_DEFINITION.name: SAPN2022_DEFINITION,
    SOLAR_ANALYTICS_DEFINITION.name: SOLAR_ANALYTICS_DEFINITION,
}


def _dataset_definition(dataset):
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string.")
    dataset_key = dataset.strip().lower()
    try:
        return DATASET_DEFINITIONS[dataset_key]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_DEFINITIONS))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Supported datasets: {supported}."
        ) from error


def main(dataset):
    """Run the shared conformance workflow for one named dataset."""
    definition = _dataset_definition(dataset)
    inputs = definition.load_inputs()
    prepare_dataset_site = partial(
        prepare_site,
        inputs=inputs,
        definition=definition,
    )
    results = run_conformance(
        candidate_site_ids=inputs["candidate_site_ids"],
        prepare_site=prepare_dataset_site,
        generate_site_plots=GENERATE_SITE_PLOTS_DEFAULT,
        plot_no_eligible_timestamp_days=PLOT_NO_ELIGIBLE_TIMESTAMP_DAYS,
        site_plot_dir=definition.output_dir / "overall_site_plots",
    )
    write_outputs(
        results,
        definition.output_dir,
        excluded_day_schema=definition.excluded_day_schema,
    )
    if GENERATE_METHOD_COMPARISON_PLOTS:
        generate_method_plots(
            output_dir=definition.output_dir,
            prepare_site=prepare_dataset_site,
            coverage_threshold=definition.coverage_threshold,
        )

    print("Saved outputs to", definition.output_dir)
    skipped = results["skipped_sites"]
    print(
        "Skipped (site metadata rows != 1):",
        len(skipped["not_single_inverter"]),
    )
    print("Skipped (>3 PV circuits):", len(skipped["more_than_3_pv_circuits"]))
    print("Skipped (no pv_site_net circuits):", len(skipped["no_pv_site_net"]))
    print("Skipped (no day data):", len(skipped["no_day_data"]))
    print("Skipped (no eligible days):", len(skipped["no_eligible_days"]))
    return results


# verify if the following 
# from main_run_conformance import main

# results = main(dataset="solar_analytics")
results = main(dataset="sapn2022")
