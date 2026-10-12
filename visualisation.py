import datetime
import json
import os
import random
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import timedelta
from glob import glob
from itertools import cycle
from time import sleep

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from colorama import Fore
from IPython.display import HTML, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import date2num
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")


class my_plot4:
    def __init__(
        self,
        t0,
        t1,
        df,
        plt_config,
        ax_digit,
        x_label,
        y_labels,
        x_format="%a-%m-%d %H:%M",
        group_attr="groupid",
        time_attr="time",
        E2P_attr=None,
        plot_total=False,
        plot_total_func=["sum", "mean"],
        plot_period=False,
        cmap="plasma",
        legend_loc="center right",
        num_ticks=12,
        num_yticks=12,
        save_as="",
        figsize=[12, 1],
        fontsize=10,
        color_nights=True,
        fontname="times new roman",
        kW2MW_attr=[],
        color_by="attribute",
        gridwidth=[1, 1],
        dpi=150,
        hspace=0.25,
        special_legend=[],
        special_group=[],
        title_i=0,
        title_j=None,
        MW=False,
        same_scale=True,
        step=False,
        show_legends=True,
        rotation=45,
        title="Date",
        legend_join="-",
        only1title=False,
        onlyntime=False,
        lim_legend=None,
        legend_i=None,
        legend_j=None,
        bbox_inches="tight",
        show=True,
    ):
        self.t0_lc_str = t0
        self.t1_lc_str = t1
        self.t0_lc_dt = pd.to_datetime(t0, format="%Y-%m-%d %H:%M:%S%f%z")
        self.t1_lc_dt = pd.to_datetime(t1, format="%Y-%m-%d %H:%M:%S%f%z")
        self.df = df.copy()
        self.only1title = only1title
        self.onlyntime = onlyntime
        self.lim_legend = lim_legend
        self.rotation = rotation
        self.x_format = x_format
        self.time_attr = time_attr
        self.x_label = x_label
        self.y_labels = y_labels
        self.df = (
            self.df.query(
                f"{self.time_attr} >= '{self.t0_lc_str}' and {self.time_attr} < '{self.t1_lc_str}'"
            )
            .sort_values(by=self.time_attr)
            .reset_index(drop=True)
        )
        self.plt_period = plot_period
        self.plt_attr = list(plt_config.keys())
        self.group_attr = group_attr
        self.plt_t = plot_total
        self.plt_t_func = plot_total_func
        self.plt_config = plt_config
        self.lgnd_loc = legend_loc
        self.legend_i = legend_i
        self.legend_j = legend_j
        self.bbox_inches = bbox_inches
        self.spcl_lgnd = special_legend
        self.plt_loc = np.asarray(
            np.asarray(list(plt_config.values()))[:, :2], dtype="float32"
        )
        self.n_ticks = num_ticks
        self.n_yticks = num_yticks
        self.save_path = save_as
        self.spcl_g = special_group
        self.hspace = hspace
        self.title = title
        self.legend_join = legend_join
        self.font = {"family": fontname, "weight": "normal", "size": fontsize}
        self.dpi = dpi
        self.gridwidth = gridwidth
        self.color_by = color_by
        self.title_i = title_i
        self.title_j = title_j
        self.figsize = figsize
        self.color_nights = color_nights
        self.show_legends = show_legends
        self.E2P_attr = E2P_attr
        if E2P_attr is not None:
            self.check_timedelta_E2P(MW)
        elif MW:
            self.df.loc[:, kW2MW_attr] /= 1000
        self.g_titles = self.get_g_titles()
        cmap = matplotlib.colormaps[cmap]
        # cmap = remove_yellow(cmap)
        if self.color_by == "group":
            self.colors = cmap(np.linspace(0, 1, len(self.g_titles)))
        else:
            self.colors = cmap(
                np.linspace(0, 1, len(self.g_titles) * len(self.plt_attr))
            )
            # self.colors = cmap(np.linspace(0, 1, len(self.g_titles)*len(self.plt_attr)))
        if self.plt_t:
            df_tot = (
                self.df.groupby(self.time_attr)
                .agg(
                    {
                        self.plt_attr[i]: self.plt_t_func[i]
                        for i in range(len(self.plt_attr))
                    }
                )
                .reset_index()
            )
            df_tot[self.group_attr] = "All"
            self.df = pd.concat(
                [self.df, df_tot], axis=0, ignore_index=True
            ).sort_values(by=self.time_attr)
        self.k_total = self.get_k_period()
        y_min = self.plt_config[self.plt_attr[0]][3]
        if y_min is None:
            self.y_max = self.get_y_max(same_scale)
            self.y_min = self.get_y_min(same_scale)
            if self.n_yticks is not None:
                self.yticks = self.get_yticks(same_scale)
            else:
                self.yticks = None
        c = [self.time_attr] + self.plt_attr
        self.df_grouped = (
            self.df.groupby(["num_periods", self.group_attr])
            .agg({i: lambda x: x.to_list() for i in c})
            .sort_values(by=["num_periods", self.group_attr], ascending=False)
            .reset_index()
        )
        self.fig, self.spec, self.axs, self.twin_axs = self.my_fig()
        self.step = step
        self.ax_digit = ax_digit

    def get_g_titles(self):
        group_title = self.df[self.group_attr].unique().tolist()
        group_title = sorted(group_title)
        if len(self.spcl_g) > 0:
            group_title = self.spcl_g
        elif self.plt_t:
            group_title = ["All"] + group_title
        return group_title

    def get_k_period(self):
        if not (
            isinstance(self.plt_period, np.timedelta64) or self.plt_period is False
        ):
            raise ValueError("The plot_period is not of type np.timedelta64 or False")
        if self.plt_period is False:
            k_total = 1
            self.plt_period = self.t1_lc_dt - self.t0_lc_dt
            self.df["num_periods"] = 1
        else:
            t0 = self.df[self.time_attr].max() + self.plt_period
            self.df["num_periods"] = (t0 - self.df[self.time_attr]) / self.plt_period
            self.df["num_periods"] = self.df["num_periods"].astype(int)
            k_total = self.df["num_periods"].unique().shape[0]
            # k_total = (self.t1_lc_dt - self.t0_lc_dt)/self.plt_period
        return np.ceil(k_total).astype(int) * np.unique(self.plt_loc[:, 0]).shape[0]

    def get_y_max(self, same_scale):
        y_max = None
        if same_scale:
            if self.plt_t:
                # y_max = self.df.groupby(self.time_attr)[self.plt_attr].agg({
                #     self.plt_attr[i]: self.plt_t_func[i] for i in range(len(self.plt_attr))}).max().tolist()
                y_max = (
                    self.df[self.df[self.group_attr] == "All"][self.plt_attr]
                    .max()
                    .tolist()
                )
            else:
                y_max = self.df[self.plt_attr].max().tolist()
            y_max = [i for i in y_max]
            y_loc = [(i, j) for i, j in zip(self.plt_loc[:, 0], self.plt_loc[:, 1])]
            ymax_dict = defaultdict(float)
            for loc, val in zip(y_loc, y_max):
                if loc not in ymax_dict:
                    ymax_dict[loc] = val
                else:
                    ymax_dict[loc] = max(ymax_dict[loc], val)

            # Step 2: Reassign y_max so each entry reflects the max at its location
            y_max = [ymax_dict[loc] + 0.1 for loc in y_loc]

        return y_max

    def get_y_min(self, same_scale):
        y_min = None
        if same_scale:
            if self.plt_t:
                y_min = (
                    self.df[self.df[self.group_attr] == "All"][self.plt_attr]
                    .min()
                    .tolist()
                )
            else:
                y_min = self.df[self.plt_attr].min().tolist()
            y_min = [i for i in y_min]
            y_loc = [(i, j) for i, j in zip(self.plt_loc[:, 0], self.plt_loc[:, 1])]
            y_min_dict = defaultdict(float)
            for loc, val in zip(y_loc, y_min):
                if loc not in y_min_dict:
                    y_min_dict[loc] = val
                else:
                    y_min_dict[loc] = min(y_min_dict[loc], val)

            # Step 2: Reassign y_max so each entry reflects the max at its location
            y_min = [y_min_dict[loc] - 0.1 for loc in y_loc]
        return y_min

    def get_yticks(self, same_scale):
        if same_scale:
            yticks = [
                np.linspace(self.y_min[i], self.y_max[i], self.n_yticks)
                for i in range(len(self.y_max))
            ]
        else:
            yticks = None
        return yticks

    def check_timedelta_E2P(self, MW):
        q_timedelta = (
            self.df.groupby(self.group_attr)
            .agg({self.time_attr: lambda x: x.diff().dropna().unique()})
            .reset_index(drop=False)
        )
        q_timedelta = np.unique(q_timedelta[self.time_attr].values.all())
        kwh_to_kw = (np.timedelta64(1, "h") / q_timedelta[0]).astype(int)
        print(kwh_to_kw)
        q_timedelta = [i.astype("int64") / (10**9 * 60) for i in q_timedelta]
        if len(q_timedelta) > 1:
            warnings.warn(f"different time resolution (minute): {q_timedelta}")
        # else:
        #     print(Fore.GREEN+f" time resolution (minute): {q_timedelta}")
        self.df.loc[:, self.E2P_attr] *= kwh_to_kw
        if MW:
            self.df.loc[:, self.E2P_attr] /= 1000

    def my_fig(self, ncols=1):
        matplotlib.rc("font", **self.font)
        fig = plt.figure(
            constrained_layout=True,
            figsize=[ncols * self.figsize[0], self.k_total * self.figsize[1]],
            dpi=self.dpi,
        )
        spec = GridSpec(self.k_total, ncols, figure=fig)
        axs = [0] * self.k_total
        twin_ax = [0] * self.k_total
        for k in range(self.k_total):
            axs[k] = fig.add_subplot(spec[k, 0])
            axs[k].margins(x=0)
            axs[k].set_xlabel(self.x_label)
            if self.onlyntime and k != self.k_total - 1:
                # axs[k].set_xlabel()
                axs[k].xaxis.label.set_visible(False)
            twin_ax[k] = axs[k].twinx()
            twin_ax[k].margins(x=0)
        return fig, spec, axs, twin_ax

    def save_fig(self):
        if len(self.save_path) > 1:
            plt.ioff()
            self.fig.savefig(
                os.getcwd() + "/" + self.save_path,
                transparent=True,
                bbox_inches=self.bbox_inches,
                pad_inches=0,
                dpi=self.dpi,
            )
            print("saved as: ", os.getcwd() + "/" + self.save_path)
            plt.close(self.fig)

    def color_nights_func(self):
        for i in range(0, self.k_total, np.unique(self.plt_loc[:, 0]).shape[0]):
            t0_dt = (
                self.t0_lc_dt
                + i / (np.unique(self.plt_loc[:, 0]).shape[0]) * self.plt_period
            )
            t1_dt = t0_dt + self.plt_period
            night_t0_dt = (t0_dt - timedelta(days=1)).replace(hour=22, minute=0)
            night_t1_dt = (t1_dt - self.plt_period).replace(hour=7, minute=0)
            t0_num = date2num(t0_dt)
            t1_num = date2num(t1_dt)
            night_t0_num = date2num(night_t0_dt)
            night_t1_num = date2num(night_t1_dt)
            while night_t0_num < t1_num:
                period_0 = max(t0_num, night_t0_num)
                period_1 = min(t1_num, night_t1_num)
                if period_1 > period_0:
                    for ii in range(np.unique(self.plt_loc[:, 0]).shape[0]):
                        self.axs[i + ii].axvspan(
                            period_0,
                            period_1,
                            alpha=0.1,
                            color="black",
                            label="Night Time",
                        )
                night_t0_dt = night_t0_dt + timedelta(days=1)
                night_t0_num = date2num(night_t0_dt)
                night_t1_dt = night_t1_dt + timedelta(days=1)
                night_t1_num = date2num(night_t1_dt)

    def do(self):
        num_periods = self.df_grouped["num_periods"].unique()
        num_plots_per = np.unique(self.plt_loc[:, 0]).shape[0]
        k_period = -1
        for k in range(0, self.k_total, num_plots_per):
            if k % num_plots_per == 0:
                k_period += 1

            df_part = self.df_grouped[
                self.df_grouped["num_periods"] == num_periods[k_period]
            ]
            time_axis = df_part[self.time_attr]
            time_axis = np.asarray(time_axis)[0]
            xticks = [
                time_axis[i]
                for i in np.ceil(
                    np.linspace(0, len(time_axis) - 1, self.n_ticks, endpoint=True)
                ).astype(int)
            ]
            xlabels = [
                time_axis[i].strftime(self.x_format)
                for i in np.ceil(
                    np.linspace(0, len(time_axis) - 1, self.n_ticks, endpoint=True)
                ).astype(int)
            ]
            j_color = 0
            for g_title in self.g_titles:
                # for  plt_attr in self.plt_attr:
                for plt_attr in self.plt_attr:
                    # for g_title in self.g_titles:
                    y_axis = df_part[df_part[self.group_attr] == g_title][plt_attr]
                    y_axis = np.asarray(y_axis)[0]
                    fig_num = self.plt_config[plt_attr][0]
                    ax_loc = self.plt_config[plt_attr][1]
                    linestyle = self.plt_config[plt_attr][2]
                    y_min = self.plt_config[plt_attr][3]
                    y_max = self.plt_config[plt_attr][4]
                    if len(self.plt_config[plt_attr]) > 5:
                        linecolor = self.plt_config[plt_attr][5]
                    else:
                        linecolor = self.colors[j_color]
                    if y_min is not None:
                        yticks = np.linspace(y_min, y_max, self.n_yticks)
                    elif self.yticks is None:
                        yticks = np.linspace(min(y_axis), max(y_axis), self.n_yticks)
                    else:
                        yticks = self.yticks[self.plt_attr.index(plt_attr)]
                    if ax_loc == 0:
                        ax = self.axs[k + fig_num]
                        if self.step:
                            ax.step(
                                time_axis,
                                y_axis,
                                where="post",
                                color=linecolor,
                                linestyle=linestyle,
                            )
                        else:
                            ax.plot(
                                time_axis, y_axis, color=linecolor, linestyle=linestyle
                            )
                        ax.grid(linewidth=self.gridwidth[0])
                        if self.onlyntime:
                            if ax == self.axs[-1]:
                                ax.set_xticklabels(
                                    xlabels, rotation=self.rotation, ha="right"
                                )
                            else:
                                ax.set_xticklabels(
                                    [], rotation=self.rotation, ha="right"
                                )
                        else:
                            ax.set_xticklabels(
                                xlabels, rotation=self.rotation, ha="right"
                            )
                    else:
                        ax = self.twin_axs[k + fig_num]
                        if self.step:
                            ax.step(
                                time_axis,
                                y_axis,
                                where="post",
                                color=linecolor,
                                linestyle=linestyle,
                            )
                        else:
                            ax.plot(
                                time_axis, y_axis, color=linecolor, linestyle=linestyle
                            )
                        ax.grid(linewidth=self.gridwidth[1], linestyle=":")
                    t0_dt = time_axis[0]
                    t1_dt = time_axis[-1]

                    if self.only1title:
                        if ax == self.axs[0]:
                            if t0_dt.date() != t1_dt.date():
                                if (
                                    len(str(t0_dt.date())[self.title_i : self.title_j])
                                    > 0
                                ):
                                    ax.set_title(
                                        f"{self.title} {str(t0_dt.date())[self.title_i : self.title_j]}---{str(t1_dt.date())[self.title_i : self.title_j]}"
                                    )
                            else:
                                ax.set_title(
                                    f"{self.title} {str(t0_dt.date())[self.title_i : self.title_j]}"
                                )
                    else:
                        if t0_dt.date() != t1_dt.date():
                            if len(str(t0_dt.date())[self.title_i : self.title_j]) > 0:
                                ax.set_title(
                                    f"{self.title} {str(t0_dt.date())[self.title_i : self.title_j]}---{str(t1_dt.date())[self.title_i : self.title_j]}"
                                )
                        else:
                            ax.set_title(
                                f"{self.title} {str(t0_dt.date())[self.title_i : self.title_j]}"
                            )

                    ax.set_ylabel(self.y_labels[self.plt_attr.index(plt_attr)])
                    ax.set_xlim(min(time_axis), max(time_axis))
                    ax.set_xticks(xticks)
                    ax.set_yticks(yticks)

                    ax.yaxis.set_major_formatter(FuncFormatter(self.format_ticks))
                    if y_min is not None:
                        ax.set_ylim(y_min, y_max)
                    elif self.y_max is not None:
                        ax.set_ylim(
                            self.y_min[self.plt_attr.index(plt_attr)],
                            self.y_max[self.plt_attr.index(plt_attr)],
                        )
                    if self.color_by == "attribute":
                        j_color += 1
                if self.color_by == "group":
                    j_color += 1

        if self.show_legends:
            self.set_legends()
        if self.color_nights:
            self.color_nights_func()
        self.save_fig()

    def set_legends(self):
        self.lgnd_loc = cycle(self.lgnd_loc)
        plt_attr = np.char.array(self.plt_attr)
        for k in range(0, self.k_total, np.unique(self.plt_loc[:, 0]).shape[0]):
            for i in range(np.unique(self.plt_loc[:, 0]).shape[0]):
                ax_0_idx = np.logical_and(
                    self.plt_loc[:, 0] == i, self.plt_loc[:, 1] == 0
                )
                ax_1_idx = np.logical_and(
                    self.plt_loc[:, 0] == i, self.plt_loc[:, 1] == 1
                )
                lgnds_0 = plt_attr[ax_0_idx]
                # print(k, i, lgnds_0)
                lgnds_0 = [ll[self.legend_i : self.legend_j] for ll in lgnds_0]
                lgnds_1 = plt_attr[ax_1_idx]
                # print(k, i, lgnds_1)
                lgnds_1 = [ll[self.legend_i : self.legend_j] for ll in lgnds_1]
                if len(lgnds_0) > 0:
                    self.axs[k + i].legend(
                        [
                            str(g_title) + self.legend_join + str(lgnd)
                            for g_title in self.g_titles
                            for lgnd in lgnds_0
                        ],
                        loc=next(self.lgnd_loc),
                    )
                    # self.axs[k+i].grid(linewidth=self.gridwidth[0])
                else:
                    self.axs[k + i].set_yticks([])
                if len(lgnds_1) > 0:
                    # pass
                    legend_vec = [
                        str(g_title) + self.legend_join + str(lgnd)
                        for g_title in self.g_titles
                        for lgnd in lgnds_1
                    ]
                    if self.lim_legend is not None:
                        lines = self.twin_axs[k + i].get_lines()
                        lines[-1].set_label(self.lim_legend)
                        self.twin_axs[k + i].legend(loc=self.lgnd_loc[1])
                    else:
                        self.twin_axs[k + i].legend(legend_vec, loc=next(self.lgnd_loc))
                    # self.twin_axs[k+i].grid(linewidth=self.gridwidth[1], linestyle=':')
                else:
                    self.twin_axs[k + i].set_yticks([])

    def format_ticks(self, x, pos):
        string_format = "{:0" + self.ax_digit + "}"
        return string_format.format(x)


def remove_yellow(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, cmap.N))
    # Remove yellow by setting its RGB values to NaN or another color
    for i, color in enumerate(colors):
        if np.allclose(color[:3], [1, 1, 0], atol=0.1):  # Adjust tolerance as needed
            colors[i] = [1, 0.65, 0, 1]  # Replace yellow with black or another color
            # colors[i] = None  # Replace yellow with black or another color
    return LinearSegmentedColormap.from_list("custom_cmap", colors)
