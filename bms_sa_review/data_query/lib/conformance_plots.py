"""Small reusable plots for notebooks 02 and 03."""

import math

import matplotlib.pyplot as plt
import numpy as np


PURPLE = "#6d28d9"
ORANGE = "#ea580c"
GREEN = "#15803d"
GRID = "#e5e7eb"


def plot_fleet_summary(summary):
    d = summary.dropna(subset=["fleet_conformant_pct"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    bars = ax.barh(d["metric"], d["fleet_conformant_pct"], color=PURPLE)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xlim=(0, 100), xlabel="Sites classified conformant (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_site_distributions(panels, threshold=0.10):
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6), dpi=130)
    axes = np.atleast_1d(axes)
    for ax, (label, frame) in zip(axes, panels):
        ax.hist(frame["nonconf_frac"].clip(0, 1) * 100, bins=40,
                color=PURPLE, alpha=.85)
        ax.axvline(threshold * 100, color=ORANGE, ls="--",
                   label=f"Project threshold: {threshold:.0%}")
        ax.set(title=label, xlabel="Nonconforming intervals per site (%)", ylabel="Sites")
        ax.grid(color=GRID)
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_group_breakdown(frame, group_col, title):
    d = frame.sort_values("fleet_conformant_pct")
    fig, ax = plt.subplots(figsize=(9, max(3, .35 * len(d))), dpi=130)
    bars = ax.barh(d[group_col].astype(str), d["fleet_conformant_pct"], color=PURPLE)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set(xlim=(0, 100), xlabel="Sites classified conformant (%)", title=title)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_monthly_rates(*series):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
    for label, frame in series:
        ax.plot(frame["period"], frame["interval_nonconf_pct"], marker="o", label=label)
    ax.set(ylabel="Nonconforming intervals (%)", xlabel="AEST month")
    ax.grid(color=GRID)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_vvar_bands(frame):
    colors = ["#b91c1c", "#ea580c", "#f59e0b", "#65a30d", "#0284c7"]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    bars = ax.barh(frame["band"], frame["pct_of_capability_assessable"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xlabel="Capability-assessable intervals (%)",
           title="Volt-VAr response categories")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_sensitivity(frame, title):
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=130)
    ax1.plot(frame["minimum_intervals"], frame["fleet_conformant_pct"],
             marker="o", color=PURPLE)
    ax1.set_xscale("log")
    ax1.set(xlabel="Minimum evaluated intervals per site",
            ylabel="Sites classified conformant (%)", title=title)
    ax1.grid(color=GRID)
    plt.tight_layout()
    return fig
