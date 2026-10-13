"""
split_view.py
Shared look and drawing for the "where the network breaks" views.
Used by attack_views.py and adaptive_attack.py: keep this file in the same folder.

TO CHANGE HOW THE VIEWS LOOK, EDIT ONLY THE "LOOK" BLOCK BELOW.
The same settings drive the PNG figures and the Cytoscape style.
"""
import os
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# LOOK  (one column per protein group; edit values, keep the order)
# =============================================================================
FATES       = ["critical protein", "previously removed", "splits off", "stays in largest part", "already disconnected"]
FATE_COLORS = ["#F0E442",          "#000000",            "#D55E00",    "#A6CEE3",               "#E5E5E5"]
FATE_SHAPES = ["DIAMOND",          "ELLIPSE",            "ELLIPSE",    "ELLIPSE",               "ELLIPSE"]
FATE_SIZES  = [80,                 30,                   40,           25,                      18]    # Cytoscape px
FATE_BORDER = [4,                  0,                    1.5,          0,                       0]     # black border width
LABEL_SIZES = [26,                 14,                   12,           0,                       0]     # 0 = no labels
MAX_LABELS_PER_GROUP = 40          # a group bigger than this gets no labels (avoids clutter);
                                   # the critical protein is always labelled

EDGE_STATUSES = ["cut by critical protein", "inside split-off part", "lost earlier", "other"]
EDGE_COLORS   = ["#000000",                 "#D55E00",               "#BDBDBD",      "#D9D9D9"]
EDGE_WIDTHS   = [2.5,                       2.0,                     0.8,            0.8]
EDGE_LINES    = ["EQUAL_DASH",              "SOLID",                 "DOT",          "SOLID"]     # Cytoscape line types

LABEL_POSITION = "N,S,c,0.00,0.00"   # Cytoscape: label above the node (so black nodes keep readable labels)
STYLE_NAME = "Attack split"
# =============================================================================

_MPL_LINE = {"SOLID": "solid", "EQUAL_DASH": "dashed", "LONG_DASH": "dashed", "DOT": "dotted", "DASH_DOT": "dashdot"}
_MPL_SHAPE = {"DIAMOND": "D", "ELLIPSE": "o", "RECTANGLE": "s", "TRIANGLE": "^", "OCTAGON": "8", "HEXAGON": "h"}


def _look(values):
    return dict(zip(FATES, values))


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

    counts = Counter(fate.values())
    label_size = _look(LABEL_SIZES)
    label = {n: sym(n) if label_size[fate[n]] > 0 and
             (fate[n] == "critical protein" or counts[fate[n]] <= MAX_LABELS_PER_GROUP) else ""
             for n in G}

    split_nodes = set().union(*pieces[1:]) if len(pieces) > 1 else set()

    def edge_status(a, b):
        if a in removed or b in removed:
            return "lost earlier"
        if critical in (a, b):
            return "cut by critical protein"
        if a in split_nodes and b in split_nodes:
            return "inside split-off part"
        return "other"

    nodes = pd.DataFrame({"id": list(G.nodes())})
    nodes["symbol"] = nodes["id"].map(sym)
    nodes["fate"] = nodes["id"].map(fate)
    nodes["piece"] = nodes["id"].map(piece)
    nodes["label"] = nodes["id"].map(label)
    edges = pd.DataFrame([(str(a), str(b), edge_status(a, b)) for a, b in G.edges()],
                         columns=["source", "target", "edge_status"])
    return dict(G=G, critical=critical, removed_before=list(removed_before), fate=fate, label=label,
                pieces=pieces, nodes=nodes, edges=edges, lcc_before=len(main), lcc_after=lcc_after,
                counts=counts)


def make_layout(G):
    """Largest component drawn by path distance; small components in rows underneath."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.kamada_kawai_layout(G.subgraph(comps[0]))
    per_row, dx, dy = 12, 2.0 / 11, 0.16
    for i, c in enumerate(comps[1:]):
        cx, cy = -1 + (i % per_row) * dx, -1.25 - (i // per_row) * dy
        for n, (px, py) in nx.circular_layout(G.subgraph(c), scale=0.035 + 0.008 * len(c)).items():
            pos[n] = (cx + px, cy + py)
    return pos


def _mpl_size(cyto_size):
    return (cyto_size / 35.0) ** 2 * 60


def draw_view(v, pos, step, attack_name, path):
    """Two panels: intact network with targets marked | network at the breaking point."""
    G, fate, crit = v["G"], v["fate"], v["critical"]
    color, shape, size, border, lsize = (_look(FATE_COLORS), _look(FATE_SHAPES), _look(FATE_SIZES),
                                         _look(FATE_BORDER), _look(LABEL_SIZES))
    ecolor, ewidth, eline = (dict(zip(EDGE_STATUSES, EDGE_COLORS)), dict(zip(EDGE_STATUSES, EDGE_WIDTHS)),
                             dict(zip(EDGE_STATUSES, EDGE_LINES)))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    lift = 0.022 * max(max(xs) - min(xs), max(ys) - min(ys))       # labels sit just above their node

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # ---- left: intact network, attack targets marked
    gone = set(v["removed_before"])
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="#D9D9D9", width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G if n not in gone and n != crit],
                           node_color="#A6CEE3", node_size=_mpl_size(25), ax=ax1, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=list(gone), node_color=color["previously removed"],
                           node_size=_mpl_size(size["previously removed"]), ax=ax1, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=[crit], node_color=color["critical protein"],
                           node_shape=_MPL_SHAPE.get(shape["critical protein"], "o"),
                           node_size=_mpl_size(size["critical protein"]), ax=ax1,
                           edgecolors="black", linewidths=border["critical protein"] * 0.5)
    ax1.set_title(f"Intact network ({G.number_of_nodes()} proteins): the first {step} {attack_name} targets",
                  fontsize=13)
    ax1.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=8, mfc=color["previously removed"], mec="none",
               label=f"removed before step {step} ({len(gone)})"),
        Line2D([], [], ls="", marker=_MPL_SHAPE.get(shape["critical protein"], "o"), ms=12,
               mfc=color["critical protein"], mec="black", label=f"critical protein: {v['label'][crit]}")],
        loc="upper left", fontsize=11, frameon=False)

    # ---- right: breaking point
    for st in ["other", "lost earlier", "inside split-off part", "cut by critical protein"]:
        el = [(a, b) for a, b, s in v["edges"].itertuples(index=False) if s == st]
        if el:
            nx.draw_networkx_edges(G, pos, edgelist=el, ax=ax2, edge_color=ecolor[st], width=ewidth[st],
                                   style=_MPL_LINE.get(eline[st], "solid"))
    for f in ["already disconnected", "stays in largest part", "splits off", "previously removed",
              "critical protein"]:
        nl = [n for n in G if fate[n] == f]
        if nl:
            nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=color[f], ax=ax2,
                                   node_shape=_MPL_SHAPE.get(shape[f], "o"), node_size=_mpl_size(size[f]),
                                   edgecolors="black" if border[f] > 0 else color[f],
                                   linewidths=border[f] * 0.5 if border[f] > 0 else 0)
    for f in FATES:
        lab = {n: v["label"][n] for n in G if fate[n] == f and v["label"][n]}
        if lab:
            nx.draw_networkx_labels(G, {n: (pos[n][0], pos[n][1] + lift) for n in lab}, labels=lab, ax=ax2,
                                    font_size=lsize[f] * 0.5, font_weight="bold",
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))
    handles = [Line2D([], [], ls="", marker=_MPL_SHAPE.get(shape[f], "o"), ms=12 if f == "critical protein" else 8,
                      mfc=color[f], mec="black" if border[f] > 0 else color[f],
                      label=f"{f} ({v['counts'].get(f, 0)})") for f in FATES]
    handles += [Line2D([], [], color=ecolor[s], lw=ewidth[s], ls=_MPL_LINE.get(eline[s], "solid"), label=s)
                for s in EDGE_STATUSES if s != "other"]
    ax2.legend(handles=handles, loc="upper left", fontsize=11, frameon=False)
    ax2.set_title(f"Removing {v['label'][crit]} at step {step}: largest component "
                  f"{v['lcc_before']} → {v['lcc_after']} proteins", fontsize=13)
    for ax in (ax1, ax2):
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


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
            e_df = v["edges"].copy()
            e_df["interaction"] = "pp"
            suid = p4c.create_network_from_data_frames(nodes=v["nodes"].copy(), edges=e_df,
                                                       title=title, collection=collection)
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
                            "NODE_LABEL_POSITION": LABEL_POSITION, "EDGE_WIDTH": 0.8, "EDGE_TRANSPARENCY": 255}
                wanted = [
                    ("node label", "label", "p", [], []),
                    ("node fill color", "fate", "d", FATES, FATE_COLORS),
                    ("node size", "fate", "d", FATES, FATE_SIZES),
                    ("node shape", "fate", "d", FATES, FATE_SHAPES),
                    ("node border width", "fate", "d", FATES, FATE_BORDER),
                    ("node label font size", "fate", "d", FATES, LABEL_SIZES),
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
