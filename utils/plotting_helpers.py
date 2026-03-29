from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

from utils.inference_helpers import (
    build_inference_context,
    get_available_treatments,
    run_inference_for_scenario,
    summarize_results,
)


@dataclass(frozen=True)
class PlotConfig:
    truth_color: str = "red"        # for ground truth
    dssat_color: str = "blue"       # for the DSSAT simulation target
    base_color: str = "orange"
    ft_color: str = "green"
    gt_color: str = "black"         # Alternative ground truth color for comparison plot
    rain_color: str = "tab:blue"
    napp_color: str = "purple"
    rain_alpha: float = 0.18
    y_margin: float = 0.08


# ---------------------------------------------------------------------------
# Axis limit helpers
# ---------------------------------------------------------------------------

def _compute_axis_limits(
    results: List[Dict[str, Any]],
    fallback_day_cap: int,
    extra_series: Optional[List[np.ndarray]] = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return (xlim, ylim, rain_ylim) derived from actual data."""

    # X-axis: use real DayAfterPlant range across all results
    x_min_vals, x_max_vals = [], []
    for res in results:
        t = np.asarray(res["time"])
        if len(t):
            x_min_vals.append(float(np.nanmin(t)))
            x_max_vals.append(float(np.nanmax(t)))
    if x_min_vals:
        xlim = (min(x_min_vals), max(x_max_vals))
    else:
        xlim = (0.0, float(fallback_day_cap))

    y_values = []
    rain_values = []
    for res in results:
        valid = res["valid_mask"]
        y_values.extend(np.asarray(res["pred_base"]).tolist())
        if "pred_finetuned" in res:
            y_values.extend(np.asarray(res["pred_finetuned"]).tolist())
        y_values.extend(np.asarray(res["true"])[valid].tolist())
        if "dssat_target" in res:
            dt = np.asarray(res["dssat_target"]).ravel()
            y_values.extend(dt[np.isfinite(dt)].tolist())
        rain_values.extend(np.asarray(res.get("rain", [])).tolist())

    # allow caller to add extra y-series (e.g. additional lines)
    if extra_series:
        for s in extra_series:
            arr = np.asarray(s).ravel()
            y_values.extend(arr[np.isfinite(arr)].tolist())

    if y_values:
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        pad = (y_max - y_min) * 0.08
        ylim = (y_min - pad, y_max + pad)
    else:
        ylim = (0.0, 1.0)

    if rain_values:
        rain_max = float(np.nanmax(rain_values))
        if not np.isfinite(rain_max) or rain_max <= 0:
            rain_max = 1.0
    else:
        rain_max = 1.0
    rain_ylim = (0.0, rain_max * 1.10)

    return xlim, ylim, rain_ylim


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------

def _format_treatment_title(res: Dict[str, Any]) -> str:
    # If napp_sequence is provided (SoilN), use it. Otherwise use Treatment ID.
    if "napp_sequence" in res:
        return res["napp_sequence"]
    return str(res["scenario"]["Treatment"])

def _title_single(res: Dict[str, Any]) -> str:
    s = res["scenario"]
    tr_str = _format_treatment_title(res)
    return f"Year={s['Year']}, Farm={s['Farm']}, Treatment={tr_str}"

def _title_all_subplot(res: Dict[str, Any]) -> str:
    tr_str = _format_treatment_title(res)
    return f"Treatment={tr_str}"


# ---------------------------------------------------------------------------
# Core single-panel plot
# ---------------------------------------------------------------------------

def _plot_one(
    result: Dict[str, Any],
    ax: plt.Axes,
    config: PlotConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rain_ylim: tuple[float, float],
    title: str,
    show_dssat: bool = False,
    show_ft: bool = True,
    show_gt: bool = True,
    show_legend: bool = True,
) -> tuple:
    """Draw one panel. Returns (handles, labels) for shared-legend use."""
    t = np.asarray(result["time"])
    y_true = np.asarray(result["true"])
    y_base = np.asarray(result["pred_base"])
    rain = np.asarray(result.get("rain", np.zeros_like(t)))
    napp = np.asarray(result.get("napp", np.zeros_like(t)))
    valid = np.asarray(result["valid_mask"])

    # DSSAT Target
    if show_dssat and "dssat_target" in result:
        y_dssat = np.asarray(result["dssat_target"])
        valid_dssat = np.isfinite(y_dssat) & (y_dssat > 0)
        # Plot continuous line for DSSAT target by only selecting valid points
        ax.plot(t[valid_dssat], y_dssat[valid_dssat], color=config.dssat_color, label="DSSAT Target", linewidth=1.8, linestyle="-")

    # Ground Truth
    if show_gt:
        gt_color = config.gt_color if show_dssat else config.truth_color
        gt_marker = "^" if show_dssat else "o"
        
        if "smn_lb" in result and "smn_ub" in result:
            lb = np.asarray(result["smn_lb"])
            ub = np.asarray(result["smn_ub"])
            y_mean = (lb + ub) / 2.0
            yerr_lower = np.maximum(0, y_mean[valid] - lb[valid])
            yerr_upper = np.maximum(0, ub[valid] - y_mean[valid])
            ax.errorbar(t[valid], y_mean[valid], yerr=[yerr_lower, yerr_upper], 
                        fmt=gt_marker, color=gt_color, ecolor=gt_color, 
                        capsize=3, markersize=5, label="Ground Truth (Mean)", zorder=4)
        else:
            ax.scatter(t[valid], y_true[valid], color=gt_color, marker=gt_marker, label="Ground Truth", s=25, zorder=4)

    # Base model
    ax.plot(t, y_base, color=config.base_color, label="Base", linewidth=1.8)

    if show_ft and "pred_finetuned" in result:
        y_ft = np.asarray(result["pred_finetuned"])
        ax.plot(t, y_ft, color=config.ft_color, label="Fine-tuned", linewidth=1.8)

    ax.set_xlim(*xlim)
    # ax.set_ylim(0,200)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Day After Planting (DAP)", fontsize=14)
    ax.set_ylabel("kg/ha", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.35)

    ax_rain = ax.twinx()
    ax_rain.bar(t, rain, color=config.rain_color, alpha=config.rain_alpha, width=1.0, label="Rain")
    ax_rain.set_ylim(*rain_ylim)
    ax_rain.invert_yaxis()
    ax_rain.set_ylabel("Rain (mm)", fontsize=11)
    ax_rain.set_yticks([])

    # N-application: marker + value label (kg/ha)
    has_napp = np.any(napp > 0)
    if has_napp:
        y_low, y_high = ax.get_ylim()
        y_range = y_high - y_low
        y_marker = y_low + 0.07 * y_range
        y_text   = y_low + 0.13 * y_range
        event_idx = np.where(napp > 0)[0]
        for idx in event_idx:
            ax.scatter([t[idx]], [y_marker], marker="v", s=48,
                       color=config.napp_color, zorder=5)
            ax.text(t[idx], y_text, f"{napp[idx]:.0f}",
                    ha="center", va="bottom",
                    fontsize=10, color=config.napp_color,
                    rotation=90)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_rain.get_legend_handles_labels()

    merged_h, merged_l = [], []
    seen: set = set()
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in seen:
            seen.add(l)
            merged_h.append(h)
            merged_l.append(l)
    # NApp proxy handle (added once)
    if has_napp:
        napp_proxy = mlines.Line2D(
            [], [], color=config.napp_color, marker="v", linestyle="None",
            markersize=6, label="NApp (kg/ha)"
        )
        if "NApp (kg/ha)" not in seen:
            merged_h.append(napp_proxy)
            merged_l.append("NApp (kg/ha)")

    if show_legend and merged_h:
        ax.legend(merged_h, merged_l, loc="upper left", fontsize=7)

    return merged_h, merged_l


# ---------------------------------------------------------------------------
# Public: all-treatments grid (2x4) + shared legend
# ---------------------------------------------------------------------------

def _build_shared_legend_handles(config: PlotConfig, show_dssat: bool = False, show_ft: bool = True, show_gt: bool = True) -> list:
    """Build a fixed set of legend handles for the shared figure legend."""
    handles = []
    
    if show_dssat:
        handles.append(mlines.Line2D([], [], color=config.dssat_color, linestyle="-", linewidth=1.8, label="DSSAT Target (kg/ha)"))
        
    if show_gt:
        handles.append(mlines.Line2D([], [], color=config.gt_color if show_dssat else config.truth_color, 
                                     marker="^" if show_dssat else "o", linestyle="None",
                                     markersize=6, label="Ground Truth (Mean)" if show_dssat else "Ground Truth"))
                                     
    handles.append(mlines.Line2D([], [], color=config.base_color, linewidth=1.8, label="Base model"))
    
    if show_ft:
        handles.append(mlines.Line2D([], [], color=config.ft_color, linewidth=1.8, label="Fine-tuned"))
        
    handles.append(mpatches.Patch(color=config.rain_color, alpha=config.rain_alpha, label="Rain"))
    handles.append(mlines.Line2D([], [], color=config.napp_color, marker="v", linestyle="None", markersize=6, label="N applied (kg/ha)"))
    
    return handles


def plot_scenarios(
    mode: str,
    year: int,
    farm: str,
    treatment: Optional[Any] = None,
    scenario_filters: Optional[Dict[str, Any]] = None,
    plot_scope: str = "single",
    show_plot: bool = True,
    ncols: int = 4,           # default 4 => 2x4 for 8 treatments
    config: Optional[PlotConfig] = None,
    show_dssat: bool = False,
    show_ft: bool = True,
    show_gt: bool = True,
) -> Dict[str, Any]:
    """Plot SoilN or Tuber inference results.

    plot_scope: 'single' | 'all'
    For 'all', lays out a 4-column (2-row for 8 treatments) grid with a
    shared legend instead of per-subplot legends.
    """
    scope = str(plot_scope).strip().lower()
    if scope not in {"single", "all"}:
        raise ValueError("plot_scope must be either 'single' or 'all'.")

    cfg = config or PlotConfig()
    ctx = build_inference_context(mode=mode)

    if scope == "single":
        result = run_inference_for_scenario(
            ctx, year=year, farm=farm,
            treatment=treatment,
            scenario_filters=scenario_filters,
        )
        if show_plot:
            _, ax = plt.subplots(figsize=(9, 5))
            xlim, ylim, rain_ylim = _compute_axis_limits(
                [result], fallback_day_cap=result.get("day_cap", ctx.spec.default_day_cap)
            )
            h, l = _plot_one(
                result, ax=ax, config=cfg,
                xlim=xlim, ylim=ylim, rain_ylim=rain_ylim,
                title=_title_single(result),
                show_dssat=show_dssat, show_ft=show_ft, show_gt=show_gt,
                show_legend=True,
            )
            plt.tight_layout()
            plt.show()
        return {"results": [result], "summary": summarize_results([result])}

    # --- 'all' mode ---------------------------------------------------------
    treatments = get_available_treatments(ctx, year=year, farm=farm,
                                          scenario_filters=scenario_filters)
    if treatment is not None:
        treatments = [t for t in treatments if t == treatment]

    if not treatments:
        raise ValueError(f"No treatments found for mode={mode}, year={year}, farm={farm}.")

    results = []
    for tr in treatments:
        res = run_inference_for_scenario(
            ctx, year=year, farm=farm, treatment=tr,
            scenario_filters=scenario_filters,
        )
        results.append(res)

    if show_plot:
        n = len(results)
        nrows = math.ceil(n / ncols)
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(5.5 * ncols, 4.2 * nrows + 0.8),
                                squeeze=False)
        ax_list = list(axs.flatten())

        day_cap = results[0].get("day_cap", ctx.spec.default_day_cap)
        xlim, ylim, rain_ylim = _compute_axis_limits(results, fallback_day_cap=day_cap)

        has_gt = False  # ground-truth series removed

        for i, res in enumerate(results):
            _plot_one(
                res, ax=ax_list[i], config=cfg,
                xlim=xlim, ylim=ylim, rain_ylim=rain_ylim,
                title=_title_all_subplot(res),
                show_dssat=show_dssat, show_ft=show_ft, show_gt=show_gt,
                show_legend=False,   # shared legend instead
            )

        for j in range(n, len(ax_list)):
            ax_list[j].set_visible(False)

        # Shared legend at top of figure
        legend_handles = _build_shared_legend_handles(cfg, show_dssat=show_dssat, show_ft=show_ft, show_gt=show_gt)
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            fontsize=11 if show_dssat else 10,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 1.05 if show_dssat else 1.0),
        )
        # fig.suptitle(f"Year={year}, Farm={farm} ",
        #              fontsize=20, y=1.06 if show_dssat else 1.03 )
        plt.tight_layout()
        plt.ylim(0,200)
        plt.show()

    return {"results": results, "summary": summarize_results(results)}

#
