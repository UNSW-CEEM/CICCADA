"""
Deprecated plotting entrypoint.

Use plots/generate_method_plots.py instead and set PHASE_B_METHODS_TO_PLOT in
that script to the method set you want to review.
"""


def main():
    raise SystemExit(
        "Deprecated: use plots/generate_method_plots.py. "
        "If you want per-method plots, run it once for each method."
    )


if __name__ == "__main__":
    main()
