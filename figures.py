"""
figures.py -- every report figure, drawn from the saved result tables.

Nothing is recomputed here except what a picture needs, so you can restyle and
re-run this file in seconds, as often as you like.

  Colours, fonts, sizes  -> style.py   (one palette for every figure)
  Network positions      -> figures/network_layout.tsv  (made by network_layout.py)
  Each figure            -> one function below; edit it freely

Run order (once): static_attack.py, adaptive_attack.py, random_failure.py,
random_fractions.py, network_layout.py, attack_views.py -- then this file.
Run the WHOLE file:   %run figures.py   (notebook)   or   python figures.py
Every figure is saved in figures/ as .png (300 dpi), .pdf and .svg (editable text).
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style
import split_view as sv
importlib.reload(style)
importlib.reload(sv)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
STATIC_DIR = "attack_static"
ADAPTIVE_DIR = "attack_adaptive"
RANDOM_DIR = "attack_random"
FIG_DIR = "figures"
LAYOUT_FILE = f"{FIG_DIR}/network_layout.tsv"

MAKE = ["degree_distribution", "damage_curves", "robustness", "fixed_levels",
        "split_views", "centrality_vs_damage"]         # delete a name to skip that figure
SPLIT_VIEWS = [("adaptive", "betweenness"), ("adaptive", "degree"), ("adaptive", "closeness"),
               ("static", "betweenness"), ("static", "degree"), ("static", "closeness")]
DEGREE_DOT_SIZE = 22            # degree figure: dot area (pt^2)
DEGREE_DOT_EDGE = None          # e.g. "#256abf" for a thin darker outline; None = plain light blue
DEGREE_TITLE = "Degree distribution"   # "" for no title (e.g. when the caption says it)
CURVES_XMAX = 1.0              # e.g. 0.5 to zoom the damage curves on the first half
SPLIT_SIZE = 1.0               # split views: protein size (1 = as in the attack scripts' figures)
SPLIT_COMBINED_COLUMNS = 2      # panels per row in split_views_combined
SPLIT_COMBINED_SIZE = 1.6       # combined figure: proteins this much bigger, since each panel is small
FIXED_LEVEL_MEASURE = "largest component"   # or "efficiency"
NAMES = {"degree": "Degree", "betweenness": "Betweenness", "closeness": "Closeness"}

os.makedirs(FIG_DIR, exist_ok=True)
style.apply()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load(path, **kw):
    if not os.path.exists(path):
        print(f"  missing {path}: run the script that makes it")
        return None
    return pd.read_csv(path, sep="\t", **kw)


def finish(fig, name):
    style.save(fig, f"{FIG_DIR}/{name}")
    print(f"  saved {FIG_DIR}/{name}.png / .pdf / .svg")


def rows(handles, ncol):
    """Order legend entries to read left-to-right, row by row (matplotlib fills by column)."""
    blank = Line2D([], [], ls="", label=" ")
    nrow = -(-len(handles) // ncol)
    grid = handles + [blank] * (nrow * ncol - len(handles))
    return [grid[r * ncol + c] for c in range(ncol) for r in range(nrow)]


def attack_handles(modes=True, random=True):
    h = [Line2D([], [], color=style.STRATEGY_COLORS[s], lw=2, label=NAMES[s]) for s in style.STRATEGY_ORDER]
    if modes:
        h += [Line2D([], [], color=style.INK_SECONDARY, lw=1.3, ls=style.MODE_LINESTYLE[m],
                     label=m.capitalize()) for m in style.MODE_ORDER]
    if random:
        h += [Patch(fc=style.RANDOM_BAND, ec="none", label="Random failure, 5–95% of runs"),
              Line2D([], [], color=style.RANDOM_LINE, lw=1.2, label="Random failure, mean")]
    return h


def marker_handles():
    h = [Line2D([], [], ls="", marker=style.MODE_MARKER["adaptive"], ms=5, mfc=style.INK_SECONDARY,
                mec=style.INK_SECONDARY, label="Adaptive"),
         Line2D([], [], ls="", marker=style.MODE_MARKER["static"], ms=5, mfc="white",
                mec=style.INK_SECONDARY, label="Static")]
    return [Line2D([], [], ls="", marker="s", ms=6, mfc=style.STRATEGY_COLORS[s], mec="none", label=NAMES[s])
            for s in style.STRATEGY_ORDER] + h


# ----------------------------------------------------------------------------
# Data shared by several figures
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))


def sym(n):
    return SYM.get(n, n)


layout = load(LAYOUT_FILE)
if layout is not None:
    POS = {str(r.id): (r.x, -r.y) for r in layout.itertuples()}      # file uses Cytoscape's y-down
else:
    print("  using a temporary layout; run network_layout.py for the shared one")
    POS = sv.make_layout(G)


# ----------------------------------------------------------------------------
# Figure: degree distribution
# ----------------------------------------------------------------------------
def degree_distribution():
    """Number of proteins with each degree, log-log axes."""
    k = np.array([d for _, d in G.degree()])
    ks, counts = np.unique(k, return_counts=True)
    fig, ax = plt.subplots(figsize=(style.HALF_WIDTH * 1.25, 2.6))
    ax.scatter(ks, counts, s=DEGREE_DOT_SIZE, color=style.NETWORK_NODE, edgecolors=DEGREE_DOT_EDGE or "none",
               linewidths=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.8, counts.max() * 1.5)
    ax.set_xlabel("Degree (k)", color=style.INK)
    ax.set_ylabel("Number of proteins", color=style.INK)
    ax.tick_params(colors=style.INK, labelcolor=style.INK, which="both")
    if DEGREE_TITLE:
        ax.set_title(DEGREE_TITLE, loc="center", color=style.INK)
    finish(fig, "degree_distribution")


# ----------------------------------------------------------------------------
# Figure: damage curves (largest component and efficiency vs fraction removed)
# ----------------------------------------------------------------------------
def damage_curves():
    cur = load(f"{ADAPTIVE_DIR}/adaptive_curves.tsv")
    if cur is None:
        cur = load(f"{STATIC_DIR}/attack_curves.tsv")
        if cur is None:
            return
        cur["mode"] = "static"
    rnd = load(f"{RANDOM_DIR}/random_curves.tsv")

    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_WIDTH, 2.6), sharex=True)
    panels = [("lcc", "lcc", "Largest component (fraction of proteins)"),
              ("eff", "eff_rel", "Global efficiency (relative to intact)")]
    for ax, (kind, rkey, ylabel), letter in zip(axes, panels, "AB"):
        if rnd is not None and f"{rkey}_p05" in rnd:
            ax.fill_between(rnd["fraction_removed"], rnd[f"{rkey}_p05"], rnd[f"{rkey}_p95"],
                            color=style.RANDOM_BAND, lw=0, zorder=1)
            ax.plot(rnd["fraction_removed"], rnd[f"{rkey}_mean"], color=style.RANDOM_LINE, lw=1.1, zorder=2)
        for s in style.STRATEGY_ORDER:
            for m in style.MODE_ORDER:
                d = cur[(cur["strategy"] == s) & (cur["mode"] == m)].sort_values("step")
                if d.empty:
                    continue
                y = d["lcc"] if kind == "lcc" else d["efficiency"] / d["efficiency"].iloc[0]
                ax.plot(d["fraction_removed"], y, color=style.STRATEGY_COLORS[s], ls=style.MODE_LINESTYLE[m],
                        lw=1.5 if m == "adaptive" else 1.2, zorder=3)
        ax.set_xlim(0, CURVES_XMAX)
        ax.set_ylim(0, None)
        ax.set_xlabel("Fraction of proteins removed")
        ax.set_ylabel(ylabel)
        style.panel_label(ax, letter, x=-0.14)
    h = attack_handles()
    h = h[:3] + [Line2D([], [], ls="", label=" ")] + h[3:]          # row 1: strategies, row 2: mode + random
    fig.legend(handles=rows(h, 4), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    finish(fig, "damage_curves")


# ----------------------------------------------------------------------------
# Figure: robustness index per attack vs random failure (dumbbell)
# ----------------------------------------------------------------------------
def robustness():
    t = load(f"{RANDOM_DIR}/targeted_vs_random.tsv", index_col=0)
    r = load(f"{RANDOM_DIR}/random_runs.tsv")
    if t is None or r is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_WIDTH, 1.9), sharey=True)
    for ax, (col, xlabel), letter in zip(axes, [("R_lcc", "Robustness index R (largest component)"),
                                                ("R_efficiency", "Robustness index R$_E$ (efficiency)")], "AB"):
        lo, hi = r[col].quantile([0.05, 0.95])
        ax.axvspan(lo, hi, color=style.RANDOM_BAND, lw=0, zorder=0)
        ax.axvline(r[col].mean(), color=style.RANDOM_LINE, lw=1.1, zorder=1)
        ax.text(r[col].mean(), -0.75, "random", ha="center", va="bottom", fontsize=style.FONT_SIZE - 1,
                color=style.INK_SECONDARY)
        for i, s in enumerate(style.STRATEGY_ORDER):
            c = style.STRATEGY_COLORS[s]
            xs, xa = t.loc[f"static {s}", col], t.loc[f"adaptive {s}", col]
            ax.plot([xs, xa], [i, i], color=c, lw=1.4, zorder=2)
            ax.scatter(xs, i, marker=style.MODE_MARKER["static"], s=34, facecolor="white", edgecolor=c,
                       linewidth=1.3, zorder=3)
            ax.scatter(xa, i, marker=style.MODE_MARKER["adaptive"], s=34, color=c, zorder=3)
        ax.set_yticks(range(len(style.STRATEGY_ORDER)))
        ax.set_yticklabels([NAMES[s] for s in style.STRATEGY_ORDER])
        ax.set_ylim(len(style.STRATEGY_ORDER) - 0.5, -0.9)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, None)
        ax.set_xlabel(xlabel + "\nlower = more damaging")
        style.panel_label(ax, letter, x=-0.04 if letter == "B" else -0.2)
    fig.legend(handles=marker_handles()[3:] + [Patch(fc=style.RANDOM_BAND, ec="none",
               label="Random failure, 5–95% of runs"), Line2D([], [], color=style.RANDOM_LINE, lw=1.1,
               label="Random failure, mean")], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    finish(fig, "robustness")


# ----------------------------------------------------------------------------
# Figure: fixed removal levels (random box plots + targeted attacks)
# ----------------------------------------------------------------------------
def fixed_levels():
    runs = load(f"{RANDOM_DIR}/random_fraction_runs.tsv")
    t = load(f"{RANDOM_DIR}/targeted_by_fraction.tsv")
    if runs is None:
        return
    lcc = FIXED_LEVEL_MEASURE == "largest component"
    col = "lcc_percent_of_intact" if lcc else "efficiency_percent_of_intact"
    levels = sorted(runs["fraction_removed"].unique())
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(style.FULL_WIDTH, 2.7))
    ax.boxplot([runs.loc[runs["fraction_removed"] == f, col] for f in levels], positions=x, widths=0.55,
               whis=(5, 95), showfliers=False, patch_artist=True, zorder=1,
               boxprops=dict(facecolor=style.RANDOM_BAND, edgecolor=style.RANDOM_LINE, lw=0.8),
               medianprops=dict(color=style.RANDOM_LINE, lw=1.2),
               whiskerprops=dict(color=style.RANDOM_LINE, lw=0.8), capprops=dict(color=style.RANDOM_LINE, lw=0.8))
    if lcc:
        ax.plot(x, [100 * (1 - f) for f in levels], color=style.INK_MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    if t is not None:
        sub = t[t["measure"] == FIXED_LEVEL_MEASURE]
        offsets = dict(zip([(s, m) for s in style.STRATEGY_ORDER for m in style.MODE_ORDER],
                           np.linspace(-0.33, 0.33, 6)))
        for (s, m), off in offsets.items():
            d = sub[sub["attack"] == f"{m} {s}"].set_index("fraction_removed").reindex(levels)
            c = style.STRATEGY_COLORS[s]
            ax.scatter(x + off, d["attack_%"], marker=style.MODE_MARKER[m], s=20, zorder=3,
                       facecolor=c if m == "adaptive" else "white", edgecolor=c, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{round(100 * f)}%" for f in levels])
    ax.set_xlabel("Proteins removed")
    ax.set_ylabel("Largest component (% of intact)" if lcc else "Global efficiency (% of intact)")
    ax.set_ylim(0, 102)
    ax.grid(axis="x", visible=False)
    extra = [Patch(fc=style.RANDOM_BAND, ec=style.RANDOM_LINE, lw=0.8,
                   label="Random failure (box 25–75%, whiskers 5–95%)")]
    if lcc:
        extra.append(Line2D([], [], color=style.INK_MUTED, lw=0.9, ls=(0, (3, 2)),
                            label="Only the removed proteins lost"))
    h = marker_handles()
    h = h[:3] + [Line2D([], [], ls="", label=" ")] + h[3:] + extra
    fig.legend(handles=rows(h, 4), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    finish(fig, "fixed_levels" + ("" if lcc else "_efficiency"))


# ----------------------------------------------------------------------------
# Figure: where the network breaks (one file per attack)
# ----------------------------------------------------------------------------
def _view(mode, s):
    """Rebuild the breaking-point view from the saved split table."""
    path = f"{STATIC_DIR}/split_{s}.tsv" if mode == "static" else f"{ADAPTIVE_DIR}/split_adaptive_{s}.tsv"
    t = load(path)
    if t is None:
        return None, None
    t["id"] = t["id"].astype(str)
    shown = set(t["id"])
    removed = [n for n in G if n not in shown] + t.loc[t["fate"] == "previously removed", "id"].tolist()
    critical = t.loc[t["fate"] == "critical protein", "id"].iloc[0]
    return sv.build_view(G, removed, critical, sym), len(removed) + 1


SPLIT_POS = sv.make_layout(G)       # same layout as the attack scripts' views (see USE_LAYOUT_FILE in split_view.py)


def _aspect(v):
    xy = np.array([SPLIT_POS[n] for n in v["shown"]])
    return np.ptp(xy[:, 1]) / np.ptp(xy[:, 0])


def split_view(mode, s):
    """One panel per view, same look as the attack scripts' views, scaled to the page width."""
    v, step = _view(mode, s)
    if v is None:
        return
    w = style.FULL_WIDTH
    fig, ax = plt.subplots(figsize=(w, w * _aspect(v) + (0.3 if sv.SHOW_TITLE else 0)))
    sv.draw_panel(ax, v, SPLIT_POS, scale=w / 10 * SPLIT_SIZE,
                  title=sv.view_title(v, step, f"{mode} {s}") if sv.SHOW_TITLE else None)
    fig.tight_layout()
    finish(fig, f"split_{mode}_{s}")


def split_views():
    for mode, s in SPLIT_VIEWS:
        split_view(mode, s)


def split_views_combined():
    """All views of SPLIT_VIEWS in one figure, panels A, B, C ..."""
    views = [(mode, s, *_view(mode, s)) for mode, s in SPLIT_VIEWS]
    views = [x for x in views if x[2] is not None]
    if not views:
        return
    ncol = min(SPLIT_COMBINED_COLUMNS, len(views))
    nrow = int(np.ceil(len(views) / ncol))
    w = style.FULL_WIDTH / ncol
    aspect = max(_aspect(x[2]) for x in views)
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.FULL_WIDTH, nrow * (w * aspect + 0.3)), squeeze=False)
    for ax, (mode, s, v, step), letter in zip(axes.flat, views, "ABCDEFGHIJ"):
        sv.draw_panel(ax, v, SPLIT_POS, scale=w / 10 * SPLIT_COMBINED_SIZE)
        ax.text(0, 1.01, letter, transform=ax.transAxes, fontsize=style.PANEL_LABEL_SIZE, fontweight="bold",
                va="bottom", ha="left")
        if sv.SHOW_TITLE:
            ax.text(0.09, 1.01, f"{mode.capitalize()} {s}: {v['lcc_before']} → {v['lcc_after']}",
                    transform=ax.transAxes, fontsize=style.FONT_SIZE - 1, fontweight="bold", va="bottom", ha="left")
    for ax in list(axes.flat)[len(views):]:
        ax.set_axis_off()
    fig.tight_layout()
    finish(fig, "split_views_combined")


# ----------------------------------------------------------------------------
# Figure: centrality vs damage when removed at random
# ----------------------------------------------------------------------------
def centrality_vs_damage():
    t = load(f"{RANDOM_DIR}/node_random_table.tsv")
    if t is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(style.FULL_WIDTH, 2.1), sharey=True)
    for ax, c, letter in zip(axes, style.STRATEGY_ORDER, "ABC"):
        ax.scatter(t[c], t["mean_random_damage"], s=9, color=style.STRATEGY_COLORS[c], alpha=0.75, lw=0)
        rho = t[c].corr(t["mean_random_damage"], method="spearman")
        ax.set_title(f"Spearman ρ = {rho:.2f}", loc="right", fontsize=style.FONT_SIZE - 1,
                     fontweight="normal", color=style.INK_SECONDARY)
        ax.set_ylim(top=t["mean_random_damage"].max() * 1.22)
        ax.set_xlabel(f"{NAMES[c]} (intact network)")
        style.panel_label(ax, letter, x=-0.06 if letter != "A" else -0.3, y=1.08)
        top = t.nlargest(3, "mean_random_damage")
        style.place_labels(ax, [(r[c], r.mean_random_damage, r.symbol) for _, r in top.iterrows()],
                           fontsize=style.FONT_SIZE - 1.5, distance=4)
    axes[0].set_ylabel("Mean proteins lost from the\nlargest component when removed")
    fig.tight_layout()
    finish(fig, "centrality_vs_damage")


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
for name in MAKE:
    print(f"{name}:")
    globals()[name]()

if "ipykernel" in sys.modules:
    plt.show()
