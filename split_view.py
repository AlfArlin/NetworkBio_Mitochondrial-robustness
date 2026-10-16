"""
split_view.py
Shared look and drawing for the "where the network breaks" views.
Used by attack_views.py, adaptive_attack.py and figures.py: keep this file in the same folder.

One panel per view:
  * light blue  = stays in the largest component
  * yellow      = splits off when the critical protein goes
  * black dots  = proteins removed in earlier steps (same size as the others)
  * red diamond = the critical protein, its name written inside
  * dashed      = interactions cut by the critical protein, kept clear of other lines
Proteins that were already disconnected (and their interactions) are left out:
the full network is shown in its own figure.

TO CHANGE HOW THE VIEWS LOOK, EDIT ONLY THE "LOOK" BLOCK BELOW.
The same settings drive the PNG figures and the Cytoscape style. Colours are set in style.py.
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects
import style                       # shared palette: colours are set in style.py

# =============================================================================
# LOOK  (one column per protein group; edit values, keep the order)
# =============================================================================
FATES       = ["critical protein", "previously removed", "splits off", "stays in largest part", "already disconnected"]
FATE_COLORS = [style.CRITICAL,     style.REMOVED,        style.SPLIT,  style.STAYS,             style.DISCONNECTED]
FATE_SHAPES = ["DIAMOND",          "ELLIPSE",            "ELLIPSE",    "ELLIPSE",               "ELLIPSE"]
FATE_SIZES  = [80,                 25,                   35,           25,                      18]    # Cytoscape px (figures scale alike)
FATE_BORDER = [3,                  0,                    1,            0,                       0]     # black border width
LABEL_SIZES = [14,                 0,                    0,            0,                       0]     # 0 = no labels (Cytoscape px)
MAX_LABELS_PER_GROUP = 40          # a group bigger than this gets no labels (avoids clutter)
LABEL_ALSO = []                    # extra proteins to label by gene name, e.g. ["COX1", "SSC1"]
LABEL_ALSO_SIZE = 14

EDGE_STATUSES = ["cut by critical protein", "inside split-off part", "lost earlier", "other"]
EDGE_COLORS   = [style.INK,                 style.SPLIT_EDGE,        style.EDGE,     style.EDGE]
EDGE_WIDTHS   = [0.8,                       0.8,                     0.8,            0.8]
EDGE_LINES    = ["EQUAL_DASH",              "SOLID",                 "DOT",          "SOLID"]     # Cytoscape line types

SHOW_REMOVED_IN_VIEW = True    # previously removed proteins as black dots (their interactions are not drawn)
SHOW_DISCONNECTED = False      # False: proteins already cut off from the largest component are left out
CUT_EDGE_GAP_PT = 1.5          # white space where a dashed line meets a protein (points)
CUT_EDGE_BEND = 0.35           # how much a dashed line may bend to avoid others (0 = always straight)
CUT_EDGE_MAX_BENT = 12         # with more cut lines than this (e.g. a whole complex), all stay straight
CRITICAL_FONT_MAX = 9          # name inside the diamond: largest font tried (pt at a 10-inch-wide figure);
                               # it shrinks until the name fits, the diamond keeps its size
SHOW_TITLE = True              # one-line title with the step and the largest-component sizes
SHOW_LEGEND = False            # True: legend in the top-left corner
USE_LAYOUT_FILE = False        # False: the attack scripts' own layout (largest component by path distance);
                               # True: the shared layout from network_layout.py (figures/network_layout.tsv)
LABEL_POSITION = "C,C,c,0.00,0.00"   # Cytoscape: the label sits inside the node (the critical diamond)
STYLE_NAME = "Attack split"
LAYOUT_FILE = os.path.join("figures", "network_layout.tsv")
# =============================================================================

_MPL_LINE = {"SOLID": "solid", "EQUAL_DASH": (0, (3, 2)), "LONG_DASH": (0, (5, 2)), "DOT": "dotted",
             "DASH_DOT": "dashdot"}


def _look(values):
    return dict(zip(FATES, values))


# -----------------------------------------------------------------------------
# What happens at the breaking point
# -----------------------------------------------------------------------------
def build_view(G, removed_before, critical, sym):
    """Classify every protein and interaction for the moment just before `critical` is removed."""
    removed = set(removed_before)
    before = G.copy()
    before.remove_nodes_from(removed)
    main = max(nx.connected_components(before), key=len)
    pieces = sorted(nx.connected_components(before.subgraph(main - {critical})), key=len, reverse=True)
    after = before.copy()
    after.remove_node(critical)
    lcc_after = max((len(c) for c in nx.connected_components(after)), default=0)

    fate = {n: "already disconnected" for n in G}
    piece = {n: "" for n in G}
    for n in removed:
        fate[n] = "previously removed"
    for i, p in enumerate(pieces):
        for n in p:
            fate[n] = "stays in largest part" if i == 0 else "splits off"
            piece[n] = "" if i == 0 else f"piece {i} ({len(p)} proteins)"
    fate[critical] = "critical protein"

    def is_shown(n):
        f = fate[n]
        return ((f != "previously removed" or SHOW_REMOVED_IN_VIEW)
                and (f != "already disconnected" or SHOW_DISCONNECTED))

    shown = [n for n in G if is_shown(n)]
    counts = Counter(fate[n] for n in shown)
    group_size = _look(LABEL_SIZES)
    also = set(LABEL_ALSO)
    label, label_size = {}, {}
    for n in G:
        f = fate[n]
        if f == "critical protein" or (group_size[f] > 0 and counts[f] <= MAX_LABELS_PER_GROUP):
            label[n], label_size[n] = sym(n), group_size[f]
        elif sym(n) in also or n in also:
            label[n], label_size[n] = sym(n), LABEL_ALSO_SIZE
        else:
            label[n], label_size[n] = "", 0

    split_nodes = set().union(*pieces[1:]) if len(pieces) > 1 else set()

    def edge_status(a, b):
        if a in removed or b in removed:
            return "lost earlier"
        if critical in (a, b):
            return "cut by critical protein"
        if a in split_nodes and b in split_nodes:
            return "inside split-off part"
        return "other"

    nodes = pd.DataFrame({"id": list(G)})                    # every protein; 'shown' says which are drawn
    nodes["symbol"] = nodes["id"].map(sym)
    nodes["fate"] = nodes["id"].map(fate)
    nodes["piece"] = nodes["id"].map(piece)
    nodes["label"] = nodes["id"].map(label)
    nodes["label_size"] = nodes["id"].map(label_size).astype(int)
    nodes["shown"] = nodes["id"].map(is_shown)
    shown_set = set(shown)
    edges = pd.DataFrame([(str(a), str(b), edge_status(a, b)) for a, b in G.edges()
                          if a in shown_set and b in shown_set and edge_status(a, b) != "lost earlier"],
                         columns=["source", "target", "edge_status"])
    return dict(G=G, critical=critical, removed_before=list(removed_before), fate=fate, label=label,
                label_size=label_size, shown=shown,
                pieces=pieces, nodes=nodes, edges=edges, lcc_before=len(main), lcc_after=lcc_after,
                counts=counts)


def make_layout(G):
    """The shared layout from network_layout.py if it exists; otherwise a quick stand-in."""
    if USE_LAYOUT_FILE and os.path.exists(LAYOUT_FILE):
        t = pd.read_csv(LAYOUT_FILE, sep="\t")
        t["id"] = t["id"].astype(str)
        if set(G) <= set(t["id"]):
            return {r.id: (r.x, -r.y) for r in t.itertuples()}         # file uses Cytoscape's y-down
        print(f"  {LAYOUT_FILE} does not match this network; using a temporary layout")
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.kamada_kawai_layout(G.subgraph(comps[0]))
    per_row, dx, dy = 12, 2.0 / 11, 0.16
    for i, c in enumerate(comps[1:]):
        cx, cy = -1 + (i % per_row) * dx, -1.25 - (i // per_row) * dy
        for n, (px, py) in nx.circular_layout(G.subgraph(c), scale=0.035 + 0.008 * len(c)).items():
            pos[n] = (cx + px, cy + py)
    return pos


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def _mpl_size(cyto_size, scale=1.0):
    """Cytoscape px -> matplotlib marker area (pt^2), as in the attack scripts' figures."""
    return (cyto_size / 35.0) ** 2 * 60 * scale ** 2


def _bezier(a, b, rad, t):
    """Points on matplotlib's arc3 curve from a to b (equal-aspect axes)."""
    m = (a + b) / 2
    c = m + rad * np.array([b[1] - a[1], -(b[0] - a[0])])
    t = t[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def _choose_bends(cut, pos, crit, others, clear):
    """Bend each dashed line just enough to keep it off other dashed lines and proteins."""
    if CUT_EDGE_BEND <= 0 or len(cut) > CUT_EDGE_MAX_BENT:
        return {e: 0.0 for e in cut}
    options = sorted({0.0} | {s * r for r in np.linspace(0, CUT_EDGE_BEND, 4)[1:] for s in (1, -1)}, key=abs)
    t = np.linspace(0.15, 0.9, 16)
    pts_other = np.array([pos[n] for n in others], dtype=float) if others else np.zeros((0, 2))
    c = np.asarray(pos[crit], float)
    placed, bend = [], {}
    order = sorted(cut, key=lambda e: np.arctan2(*(np.asarray(pos[e[1]]) - c)[::-1]))
    for e in order:
        b = np.asarray(pos[e[1]], float)
        best = None
        for rad in options:
            p = _bezier(c, b, rad, t)
            cost = 0.0
            if len(pts_other):
                near = np.hypot(*(p[:, None, :] - pts_other[None, :, :]).transpose(2, 0, 1)) < clear
                near[:, [i for i, n in enumerate(others) if n == e[1]]] = False
                cost += near.any(axis=0).sum()
            for q in placed:
                cost += 2 * (np.hypot(*(p[:, None, :] - q[None, :, :]).transpose(2, 0, 1)) < clear).any(axis=1).sum()
            cost += 0.5 * abs(rad)
            if best is None or cost < best[0]:
                best = (cost, rad, p)
        bend[e] = best[1]
        placed.append(best[2])
    return bend


def _legend(ax, v, scale):
    color, size, border = _look(FATE_COLORS), _look(FATE_SIZES), _look(FATE_BORDER)
    handles = []
    for f in ["critical protein", "previously removed", "splits off", "stays in largest part"]:
        k = v["counts"].get(f, 0)
        if k:
            handles.append(Line2D([], [], ls="", marker="D" if f == "critical protein" else "o",
                                  ms=np.sqrt(_mpl_size(size[f], scale)), mfc=color[f],
                                  mec="black" if border[f] else color[f], label=f"{f} ({k})"))
    handles += [Line2D([], [], color=style.INK, lw=1.2 * scale, ls="--", label="cut by critical protein"),
                Line2D([], [], color=style.SPLIT_EDGE, lw=1.2 * scale, label="inside split-off part")]
    ax.legend(handles=handles, loc="upper left", fontsize=11 * scale, frameon=False)


def draw_panel(ax, v, pos, scale=1.0, title=None):
    """The network at the breaking point, on the given layout.

    scale = figure width in inches / 10: sizes match the attack scripts' figures
    (a 10-inch panel), so the picture looks the same at any width.
    """
    fate, crit, G = v["fate"], v["critical"], v["G"]
    fig = ax.figure
    color, size, border = _look(FATE_COLORS), _look(FATE_SIZES), _look(FATE_BORDER)
    shown = v["shown"]
    P = {n: np.asarray(pos[n], float) for n in shown}
    xy = np.array(list(P.values()))
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    pad = 0.04 * span
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_anchor("N")
    ax.apply_aspect()
    pt_per_unit = ax.get_window_extent().width * 72 / fig.dpi / np.ptp(ax.get_xlim())

    # interactions: largest part (grey) and split-off part (yellow); cut ones are drawn later
    for st in ["other", "inside split-off part"]:
        segs = [(P[a], P[b]) for a, b, z in v["edges"].itertuples(index=False) if z == st]
        if segs:
            ax.add_collection(LineCollection(segs, colors=dict(zip(EDGE_STATUSES, EDGE_COLORS))[st],
                                             linewidths=dict(zip(EDGE_STATUSES, EDGE_WIDTHS))[st] * scale, zorder=1))
    # proteins
    for f in ["stays in largest part", "splits off", "previously removed"]:
        nl = [n for n in shown if fate[n] == f]
        if nl:
            ax.scatter([P[n][0] for n in nl], [P[n][1] for n in nl], s=_mpl_size(size[f], scale), c=color[f],
                       edgecolors="black" if border[f] else color[f],
                       linewidths=border[f] * 0.5 * scale if border[f] else 0, zorder=3)
    # critical protein: same diamond as before, name inside
    s_crit = _mpl_size(size["critical protein"], scale)
    ax.scatter(*P[crit], s=s_crit, marker="D", c=color["critical protein"], edgecolors="black",
               linewidths=border["critical protein"] * 0.5 * scale, zorder=6)
    diag_pt = np.sqrt(s_crit) * np.sqrt(2)                   # tip-to-tip width of matplotlib's "D" marker
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fs = CRITICAL_FONT_MAX * scale
    while True:                                               # largest font that fits inside the diamond
        txt = ax.text(*P[crit], v["label"][crit], ha="center", va="center", fontsize=fs,
                      fontweight="bold", color=style.INK, zorder=7)
        bb = txt.get_window_extent(renderer)
        w, h = bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi
        if w + h <= 0.88 * diag_pt or fs <= 3 * scale:
            break
        txt.remove()
        fs *= 0.92

    # cut interactions: dashed, kept apart, with white space at both ends
    cut = [(crit, b if a == crit else a) for a, b, z in v["edges"].itertuples(index=False)
           if z == "cut by critical protein"]
    node_d_unit = np.sqrt(_mpl_size(size["stays in largest part"], scale)) / pt_per_unit
    bend = _choose_bends(cut, P, crit, [n for n in shown if n != crit], clear=1.2 * node_d_unit)
    lw = EDGE_WIDTHS[0] * scale
    for e in cut:
        r_partner = np.sqrt(_mpl_size(size[fate[e[1]]], scale)) / 2
        ax.add_patch(FancyArrowPatch(P[crit], P[e[1]], arrowstyle="-", connectionstyle=f"arc3,rad={bend[e]:.3f}",
                                     shrinkA=diag_pt / 2 * 0.75 + CUT_EDGE_GAP_PT * scale,
                                     shrinkB=r_partner + CUT_EDGE_GAP_PT * scale,
                                     linestyle=(0, (4, 2.5)), lw=lw, color=style.INK, zorder=5,
                                     path_effects=[patheffects.Stroke(linewidth=lw + 2.0 * scale, foreground="white"),
                                                   patheffects.Normal()]))
    if SHOW_LEGEND:
        _legend(ax, v, scale)
    if title:
        ax.set_title(title, fontsize=13 * scale, fontweight="bold", loc="left")
    return ax


def view_title(v, step, attack_name=None):
    return (f"Removing {v['label'][v['critical']]} at step {step}: largest component "
            f"{v['lcc_before']} → {v['lcc_after']} proteins")


def draw_view(v, pos, step, attack_name, path):
    """Single-panel working figure (10 inches wide), saved as PNG (used by the attack scripts)."""
    xy = np.array([pos[n] for n in v["shown"]], dtype=float)
    aspect = np.ptp(xy[:, 1]) / max(np.ptp(xy[:, 0]), 1e-9)
    fig, ax = plt.subplots(figsize=(10, 10 * min(max(aspect, 0.5), 1.6) + (0.4 if SHOW_TITLE else 0)))
    draw_panel(ax, v, pos, scale=1.0, title=view_title(v, step) if SHOW_TITLE else None)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


# -----------------------------------------------------------------------------
# Cytoscape
# -----------------------------------------------------------------------------
def send_to_cytoscape(items, collection, base_title, out_dir):
    """items: list of (network title, view dict, png file name). Prints what it did."""
    try:
        import py4cytoscape as p4c
        p4c.cytoscape_ping()
    except Exception as e:
        print("\nCytoscape not reachable, skipping it:", type(e).__name__, e)
        return []
    base = p4c.get_network_suid(base_title) if base_title in p4c.get_network_list() else None
    made, style_ready = [], False
    for title, v, png in items:
        try:
            if title in p4c.get_network_list():
                p4c.delete_network(title)
            n_df = v["nodes"][v["nodes"]["shown"]].drop(columns="shown").copy()
            e_df = v["edges"].copy()
            e_df["interaction"] = "pp"
            suid = p4c.create_network_from_data_frames(nodes=n_df, edges=e_df, title=title, collection=collection)
            made.append(title)
            print(f"Cytoscape: created '{title}'")
        except Exception as e:
            print(f"Cytoscape: could not create '{title}':", type(e).__name__, e)
            continue
        try:
            if base is not None:
                p4c.layout_copycat(base, suid)          # same positions as your original network
            else:
                p4c.layout_network("force-directed", network=suid)
        except Exception as e:
            print("  layout step skipped:", type(e).__name__, e)
        if not style_ready:
            try:
                if STYLE_NAME in p4c.get_visual_style_names():
                    p4c.delete_visual_style(STYLE_NAME)
                defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_SIZE": 25, "NODE_BORDER_PAINT": "#000000",
                            "NODE_BORDER_WIDTH": 0, "NODE_LABEL_COLOR": "#000000", "NODE_LABEL_FONT_SIZE": 12,
                            "NODE_LABEL_POSITION": LABEL_POSITION, "EDGE_WIDTH": 0.8, "EDGE_TRANSPARENCY": 255,
                            "NETWORK_BACKGROUND_PAINT": "#FFFFFF"}
                wanted = [
                    ("node label", "label", "p", [], []),
                    ("node fill color", "fate", "d", FATES, FATE_COLORS),
                    ("node size", "fate", "d", FATES, FATE_SIZES),
                    ("node shape", "fate", "d", FATES, FATE_SHAPES),
                    ("node border width", "fate", "d", FATES, FATE_BORDER),
                    ("node label font size", "label_size", "p", [], []),
                    ("edge stroke unselected paint", "edge_status", "d", EDGE_STATUSES, EDGE_COLORS),
                    ("edge width", "edge_status", "d", EDGE_STATUSES, EDGE_WIDTHS),
                    ("edge line type", "edge_status", "d", EDGE_STATUSES, EDGE_LINES),
                ]
                mappings = []
                for prop, col, kind, keys, vals in wanted:
                    try:
                        mappings.append(p4c.map_visual_property(prop, col, kind, keys, vals, network=suid))
                    except Exception as e:
                        print(f"  mapping '{prop}' skipped:", type(e).__name__, e)
                p4c.create_visual_style(STYLE_NAME, defaults, mappings)
                try:
                    p4c.lock_node_dimensions(True, style_name=STYLE_NAME)
                except Exception as e:
                    print("  could not lock node width/height:", type(e).__name__, e)
                style_ready = True
            except Exception as e:
                print("  style step skipped:", type(e).__name__, e)
        try:
            p4c.set_visual_style(STYLE_NAME, network=suid)
            p4c.fit_content(network=suid)
        except Exception as e:
            print("  applying the style failed:", type(e).__name__, e)
        try:
            p4c.export_image(os.path.abspath(os.path.join(out_dir, png)), type="PNG", network=suid,
                             overwrite_file=True)
        except Exception as e:
            print("  image export skipped:", type(e).__name__, e)
    if made:
        print(f"\nCytoscape: the new networks are in the Network panel (left side) under '{collection}'.")
    return made
