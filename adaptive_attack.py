"""
adaptive_attack.py
Adaptive targeted attacks on the yeast mitochondrial PPI network.

Adaptive = after EVERY removal the scores are recomputed on what is left of the
network, and the protein that is now top-ranked is removed next.
Strategies : degree, betweenness, closeness
Compared to: the static version of each attack, and random failure
Extra      : single-protein knockouts on the intact network
Outputs    : tables + figures in OUT_DIR, and views in Cytoscape (if it is open)

Run the WHOLE file:   %run adaptive_attack.py   (notebook)   or   python adaptive_attack.py
Expect a few minutes of running time.
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
OUT_DIR = "attack_adaptive"
N_RANDOM = 100
SEED = 42                      # same seed as static_attack.py -> same random baseline
STRATEGIES = ["degree", "betweenness", "closeness"]
COLORS = {"degree": "#D55E00", "betweenness": "#0072B2", "closeness": "#009E73"}
TOP_FRACTION = 0.10            # "early targets" = first 10% of removals
STEP_OVERRIDE = {}             # e.g. {"betweenness": 6} to view another step
USE_CYTOSCAPE = True
BASE_NETWORK_TITLE = "Yeast mitochondrial PPI"
COLLECTION = "Adaptive attack views"
STYLE_NAME = "Attack split"

FATES = ["critical protein", "stays in largest part", "splits off", "already disconnected"]
FATE_COLORS = ["#D55E00", "#0072B2", "#E69F00", "#BBBBBB"]
FATE_SIZES = [70, 35, 35, 25]

SCORE = {
    "degree": lambda H: dict(H.degree()),
    "betweenness": lambda H: nx.betweenness_centrality(H),
    "closeness": lambda H: nx.closeness_centrality(H),
}

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
N = G.number_of_nodes()

SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))


def sym(n):
    return SYM.get(n, n)


# ----------------------------------------------------------------------------
# 2. DAMAGE MEASURES
# ----------------------------------------------------------------------------
def efficiency(H):
    """Global efficiency, normalised by the ORIGINAL number of protein pairs."""
    if H.number_of_edges() == 0:
        return 0.0
    A = nx.to_scipy_sparse_array(H, weight=None, format="csr", dtype=float)
    D = shortest_path(A, method="D", unweighted=True)
    ok = np.isfinite(D) & (D > 0)
    return float((1.0 / D[ok]).sum() / (N * (N - 1)))


def measure(H):
    sizes = [len(c) for c in nx.connected_components(H)]
    return {"lcc_n": max(sizes, default=0), "lcc": max(sizes, default=0) / N,
            "components": sum(1 for s in sizes if s > 1),
            "isolated": sum(1 for s in sizes if s == 1),
            "edges": H.number_of_edges(), "efficiency": efficiency(H)}


def finish(rows):
    df = pd.DataFrame(rows)
    df["fraction_removed"] = df["step"] / N
    df["lcc_drop"] = -df["lcc_n"].diff().fillna(0).astype(int)
    return df


def run_fixed(order):
    """Static attack or random failure: the removal order is fixed in advance."""
    H = G.copy()
    rows = [{"step": 0, "removed": None, "score_when_removed": np.nan, **measure(H)}]
    for i, node in enumerate(order, start=1):
        H.remove_node(node)
        rows.append({"step": i, "removed": node, "score_when_removed": np.nan, **measure(H)})
    return finish(rows)


def run_adaptive(strategy):
    """Recompute the score on the remaining network before every removal.
    Ties: the protein with more remaining partners goes first, then ORF name."""
    H = G.copy()
    rows = [{"step": 0, "removed": None, "score_when_removed": np.nan, **measure(H)}]
    for i in range(1, N + 1):
        if H.number_of_edges() == 0:                 # only isolated proteins are left
            target, value = min(H.nodes()), 0.0
        else:
            sc = SCORE[strategy](H)
            target = min(H.nodes(), key=lambda n: (-sc[n], -H.degree(n), n))
            value = sc[target]
        H.remove_node(target)
        rows.append({"step": i, "removed": target, "score_when_removed": value, **measure(H)})
        if i % 50 == 0:
            print(f"    {strategy}: {i}/{N} removed")
    return finish(rows)


def summarise(df):
    S0, E0 = df["lcc"].iloc[0], df["efficiency"].iloc[0]
    worst = df.loc[df["lcc_drop"].idxmax()]
    return {
        "R_lcc": df["lcc"].iloc[1:].mean(),
        "R_efficiency": (df["efficiency"].iloc[1:] / E0).mean(),
        "f_half_lcc": df.loc[df["lcc"] <= S0 / 2, "fraction_removed"].min(),
        "f_half_efficiency": df.loc[df["efficiency"] <= E0 / 2, "fraction_removed"].min(),
        "largest_single_drop": int(worst["lcc_drop"]),
        "caused_by": sym(worst["removed"]),
        "at_step": int(worst["step"]),
    }


# ----------------------------------------------------------------------------
# 3. RUN: static, adaptive, random
# ----------------------------------------------------------------------------
print(f"Intact network: {N} proteins, {G.number_of_edges()} interactions")
intact = {s: SCORE[s](G) for s in STRATEGIES}
static_order = {s: sorted(G.nodes(), key=lambda n: (-intact[s][n], n)) for s in STRATEGIES}

static, adaptive = {}, {}
for s in STRATEGIES:
    print(f"  static {s} attack ...")
    static[s] = run_fixed(static_order[s])
for s in STRATEGIES:
    print(f"  adaptive {s} attack ...")
    adaptive[s] = run_adaptive(s)

print(f"  {N_RANDOM} random-failure runs ...")
nodes = list(G.nodes())
rand_runs = [run_fixed(list(rng.permutation(nodes))) for _ in range(N_RANDOM)]
rand = {m: pd.concat([r[m] for r in rand_runs], axis=1) for m in ["lcc", "efficiency", "components"]}
rand_summ = pd.DataFrame([summarise(r) for r in rand_runs])

# ----------------------------------------------------------------------------
# 4. STATISTICS
# ----------------------------------------------------------------------------
rows = {}
for s in STRATEGIES:
    rows[f"adaptive {s}"] = summarise(adaptive[s])
for s in STRATEGIES:
    rows[f"static {s}"] = summarise(static[s])
summary = pd.DataFrame(rows).T
for col in ["R_lcc", "R_efficiency"]:
    mu, sd = rand_summ[col].mean(), rand_summ[col].std()
    vals = summary[col].astype(float)
    summary[f"{col}_z_vs_random"] = (vals - mu) / sd
    summary[f"{col}_p_vs_random"] = [((rand_summ[col] <= v).sum() + 1) / (N_RANDOM + 1) for v in vals]
summary.loc["random (mean)"] = {c: rand_summ[c].mean() for c in
                                ["R_lcc", "R_efficiency", "f_half_lcc", "f_half_efficiency"]}

# what recomputing buys: static R minus adaptive R (positive = adaptive does more damage)
gain = pd.DataFrame({
    "R_lcc_static": [summary.loc[f"static {s}", "R_lcc"] for s in STRATEGIES],
    "R_lcc_adaptive": [summary.loc[f"adaptive {s}", "R_lcc"] for s in STRATEGIES],
}, index=STRATEGIES).astype(float)
gain["gain_from_adapting"] = gain["R_lcc_static"] - gain["R_lcc_adaptive"]
gain["gain_percent"] = 100 * gain["gain_from_adapting"] / gain["R_lcc_static"]

# how different are the adaptive and static removal orders?
k = max(1, round(TOP_FRACTION * N))
adaptive_order = {s: adaptive[s]["removed"].iloc[1:].tolist() for s in STRATEGIES}
adaptive_step = {s: {n: i + 1 for i, n in enumerate(adaptive_order[s])} for s in STRATEGIES}
static_rank = {s: {n: i + 1 for i, n in enumerate(static_order[s])} for s in STRATEGIES}
order_cmp = pd.DataFrame({
    "spearman_static_vs_adaptive": [pd.Series(static_rank[s]).corr(pd.Series(adaptive_step[s]), method="spearman")
                                    for s in STRATEGIES],
    f"shared_in_first_{k}": [len(set(static_order[s][:k]) & set(adaptive_order[s][:k])) for s in STRATEGIES],
}, index=STRATEGIES)
across = pd.DataFrame([[len(set(adaptive_order[a][:k]) & set(adaptive_order[b][:k])) for b in STRATEGIES]
                       for a in STRATEGIES], index=STRATEGIES, columns=STRATEGIES)

# ---- single-protein knockouts on the intact network
lcc0 = max(nx.connected_components(G), key=len)
E0 = efficiency(G)
ko = []
for n in G.nodes():
    H = G.copy()
    H.remove_node(n)
    lost = len(lcc0) - len(max(nx.connected_components(H), key=len)) - (1 if n in lcc0 else 0)
    ko.append({"id": n, "symbol": sym(n), "proteins_disconnected": lost,
               "efficiency_loss_percent": 100 * (1 - efficiency(H) / E0)})
ko = pd.DataFrame(ko).sort_values(["proteins_disconnected", "efficiency_loss_percent"], ascending=False)
n_cut = sum(1 for _ in nx.articulation_points(G))

pd.set_option("display.width", 220)
print("\n=== SUMMARY (lower R = more damaging) ===")
print(summary.round(4).to_string())
print("\n=== What recomputing the ranking buys (largest-component R) ===")
print(gain.round(4).to_string())
print("\n=== Static order vs adaptive order ===")
print(order_cmp.round(3).to_string())
print(f"\n=== Shared proteins among the first {k} adaptive targets ===")
print(across.to_string())
for s in STRATEGIES:
    top = adaptive[s].iloc[1:16]
    print(f"\n=== First 15 targets, adaptive {s} (proteins lost from largest component) ===")
    print(", ".join(f"{sym(n)} ({d})" for n, d in zip(top["removed"], top["lcc_drop"])))
print(f"\n=== Single-protein knockouts on the intact network ({n_cut} proteins are cut points) ===")
print(ko.head(10).round(2).to_string(index=False))

# ---- save tables
curves = pd.concat([df.assign(strategy=s, mode="adaptive") for s, df in adaptive.items()] +
                   [df.assign(strategy=s, mode="static") for s, df in static.items()], ignore_index=True)
curves["symbol"] = curves["removed"].map(lambda n: sym(n) if n else None)
curves.to_csv(f"{OUT_DIR}/adaptive_curves.tsv", sep="\t", index=False)
summary.to_csv(f"{OUT_DIR}/adaptive_summary.tsv", sep="\t")
gain.to_csv(f"{OUT_DIR}/adaptive_vs_static.tsv", sep="\t")
ko.to_csv(f"{OUT_DIR}/single_knockout.tsv", sep="\t", index=False)

node_table = pd.DataFrame({"id": list(G.nodes())})
node_table["symbol"] = node_table["id"].map(sym)
for s in STRATEGIES:
    a = adaptive[s].iloc[1:].set_index("removed")
    node_table[f"{s}_intact"] = node_table["id"].map(intact[s])
    node_table[f"static_rank_{s}"] = node_table["id"].map(static_rank[s])
    node_table[f"adaptive_step_{s}"] = node_table["id"].map(adaptive_step[s])
    node_table[f"adaptive_score_{s}"] = node_table["id"].map(a["score_when_removed"])
    node_table[f"adaptive_damage_{s}"] = node_table["id"].map(a["lcc_drop"])
node_table = node_table.merge(ko.drop(columns="symbol"), on="id", how="left")
node_table.to_csv(f"{OUT_DIR}/node_adaptive_table.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------
# 5. FIGURES: curves and robustness
# ----------------------------------------------------------------------------
x = adaptive[STRATEGIES[0]]["fraction_removed"]
E_int = adaptive[STRATEGIES[0]]["efficiency"].iloc[0]
panels = [("lcc", "Largest component (fraction of all proteins)", 1.0),
          ("efficiency", "Global efficiency (relative to intact)", E_int),
          ("components", "Connected groups of 2+ proteins", 1.0)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (m, label, norm) in zip(axes, panels):
    r = rand[m] / norm
    ax.fill_between(x, r.quantile(0.05, axis=1), r.quantile(0.95, axis=1), color="0.6", alpha=0.35,
                    label="random, 5-95% of runs")
    ax.plot(x, r.mean(axis=1), color="0.2", lw=1.3, ls="--", label="random, mean")
    for s in STRATEGIES:
        ax.plot(x, static[s][m] / norm, color=COLORS[s], lw=1.1, ls=":", label=f"{s}, static")
        ax.plot(x, adaptive[s][m] / norm, color=COLORS[s], lw=2.2, label=f"{s}, adaptive")
    ax.set_xlabel("Fraction of proteins removed")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"Adaptive (solid) vs static (dotted) targeted attacks: {N} proteins, {G.number_of_edges()} interactions")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/adaptive_curves.png", dpi=200)

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
pos_x = np.arange(len(STRATEGIES))
for ax, col, title in zip(axes2, ["R_lcc", "R_efficiency"],
                          ["Robustness index R (largest component)", "Robustness index (efficiency)"]):
    st = [float(summary.loc[f"static {s}", col]) for s in STRATEGIES]
    ad = [float(summary.loc[f"adaptive {s}", col]) for s in STRATEGIES]
    ax.bar(pos_x - 0.2, st, 0.4, color=[COLORS[s] for s in STRATEGIES], alpha=0.45, label="static (pale)")
    ax.bar(pos_x + 0.2, ad, 0.4, color=[COLORS[s] for s in STRATEGIES], label="adaptive (solid)")
    mu, sd = rand_summ[col].mean(), rand_summ[col].std()
    ax.axhline(mu, color="0.2", ls="--", label="random, mean")
    ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="0.6", alpha=0.3, label="random, mean ± 2 SD")
    ax.set_xticks(pos_x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/adaptive_vs_static_R.png", dpi=200)

# ----------------------------------------------------------------------------
# 6. WHERE THE NETWORK BREAKS: one view per adaptive attack
# ----------------------------------------------------------------------------
def largest_component(H):
    return max(nx.connected_components(H), key=len, default=set())


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


views = {}
for s in STRATEGIES:
    df = adaptive[s]
    step = STEP_OVERRIDE.get(s, int(df.loc[df["lcc_drop"].idxmax(), "step"]))
    removed_before = adaptive_order[s][:step - 1]
    critical = adaptive_order[s][step - 1]
    before = G.copy()
    before.remove_nodes_from(removed_before)
    main = largest_component(before)
    pieces = sorted(nx.connected_components(before.subgraph(main - {critical})), key=len, reverse=True)

    fate = {n: "already disconnected" for n in before}
    piece = {n: "" for n in before}
    for i, p in enumerate(pieces):
        for n in p:
            fate[n] = "stays in largest part" if i == 0 else "splits off"
            piece[n] = "" if i == 0 else f"piece {i} ({len(p)} proteins)"
    fate[critical] = "critical protein"

    table = pd.DataFrame({"id": list(before.nodes())})
    table["symbol"] = table["id"].map(sym)
    table["fate"] = table["id"].map(fate)
    table["piece"] = table["id"].map(piece)
    table.to_csv(f"{OUT_DIR}/split_adaptive_{s}.tsv", sep="\t", index=False)
    views[s] = dict(step=step, critical=critical, removed_before=removed_before, before=before, table=table,
                    fate=fate, lcc_before=len(main), lcc_after=int(df.loc[df["step"] == step, "lcc_n"].iloc[0]))

    print(f"\n=== ADAPTIVE {s.upper()}: most damaging removal is step {step}, {sym(critical)} ===")
    print(f"Largest component: {len(main)} -> {views[s]['lcc_after']} proteins")
    print(f"Pieces that split off: {[len(p) for p in pieces[1:]] or 'none'}")
    print(f"Removed before it ({len(removed_before)}): {', '.join(sym(n) for n in removed_before)}")
    for i, p in enumerate(pieces[1:4], start=1):
        print(f"Piece {i} ({len(p)}): {', '.join(sorted(sym(n) for n in p))}")

pos = make_layout(G)
color_of = dict(zip(FATES, FATE_COLORS))
for s, v in views.items():
    figv, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    gone = set(v["removed_before"])
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="0.75", width=0.4, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G if n not in gone and n != v["critical"]],
                           node_color="#9ecae1", node_size=22, ax=ax1, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=list(gone), node_color="black", node_size=30, ax=ax1,
                           label=f"removed earlier ({len(gone)})")
    nx.draw_networkx_nodes(G, pos, nodelist=[v["critical"]], node_color=color_of["critical protein"],
                           node_size=140, ax=ax1, label=f"critical protein: {sym(v['critical'])}")
    ax1.set_title(f"Intact network ({N} proteins) with the first {v['step']} adaptive {s} targets marked")
    ax1.legend(loc="upper left", fontsize=9)

    B = v["before"]
    nx.draw_networkx_edges(B, pos, edgelist=[e for e in B.edges() if v["critical"] not in e], ax=ax2,
                           edge_color="0.7", width=0.4, alpha=0.6)
    nx.draw_networkx_edges(B, pos, edgelist=list(B.edges(v["critical"])), ax=ax2,
                           edge_color=color_of["critical protein"], width=1.6, style="dashed")
    for f in FATES[::-1]:
        nl = [n for n in B if v["fate"][n] == f]
        nx.draw_networkx_nodes(B, pos, nodelist=nl, node_color=color_of[f], ax=ax2, linewidths=0,
                               node_size=140 if f == "critical protein" else 26, label=f"{f} ({len(nl)})")
    lab = {v["critical"]: sym(v["critical"])}
    if B.degree(v["critical"]) <= 15:
        lab.update({n: sym(n) for n in B.neighbors(v["critical"])})
    nx.draw_networkx_labels(B, pos, labels=lab, font_size=8, font_weight="bold", ax=ax2)
    ax2.set_title(f"Removing {sym(v['critical'])} (step {v['step']}): largest component "
                  f"{v['lcc_before']} -> {v['lcc_after']} proteins")
    ax2.legend(loc="upper left", fontsize=9)
    ax1.set_axis_off()
    ax2.set_axis_off()
    figv.suptitle(f"Adaptive {s} attack: where the network breaks", fontsize=14)
    figv.tight_layout()
    figv.savefig(f"{OUT_DIR}/split_adaptive_{s}.png", dpi=200)

print(f"\nTables and figures saved in '{OUT_DIR}'.")

# ----------------------------------------------------------------------------
# 7. CYTOSCAPE
# ----------------------------------------------------------------------------
if USE_CYTOSCAPE:
    try:
        import py4cytoscape as p4c
        p4c.cytoscape_ping()
        connected = True
    except Exception as e:
        connected = False
        print("\nCytoscape not reachable, skipping it:", type(e).__name__, e)

    if connected:
        base = p4c.get_network_suid(BASE_NETWORK_TITLE) if BASE_NETWORK_TITLE in p4c.get_network_list() else None
        if base is not None:
            try:        # adaptive columns on your original network (removal step, damage, knockout)
                p4c.load_table_data(node_table, data_key_column="id", table_key_column="name", network=base)
                print(f"Cytoscape: adaptive columns added to '{BASE_NETWORK_TITLE}'")
            except Exception as e:
                print("Cytoscape: could not add columns to the original network:", type(e).__name__, e)

        style_ready, made = False, []
        for s, v in views.items():
            title = f"Adaptive {s}: split at step {v['step']} ({sym(v['critical'])})"
            try:
                if title in p4c.get_network_list():
                    p4c.delete_network(title)
                B = v["before"]
                e_df = pd.DataFrame([(str(a), str(b)) for a, b in B.edges()], columns=["source", "target"])
                e_df["interaction"] = "pp"
                suid = p4c.create_network_from_data_frames(nodes=v["table"].copy(), edges=e_df,
                                                           title=title, collection=COLLECTION)
                made.append(title)
                print(f"Cytoscape: created '{title}'")
            except Exception as e:
                print(f"Cytoscape: could not create the {s} network:", type(e).__name__, e)
                continue
            try:
                if base is not None:
                    p4c.layout_copycat(base, suid)
                else:
                    p4c.layout_network("force-directed", network=suid)
            except Exception as e:
                print("  layout step skipped:", type(e).__name__, e)
            try:
                if not style_ready:
                    if STYLE_NAME in p4c.get_visual_style_names():
                        p4c.delete_visual_style(STYLE_NAME)
                    defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_SIZE": 35, "NODE_LABEL_FONT_SIZE": 9,
                                "NODE_BORDER_WIDTH": 0, "EDGE_TRANSPARENCY": 110, "EDGE_WIDTH": 1}
                    mappings = [
                        p4c.map_visual_property("node label", "symbol", "p", network=suid),
                        p4c.map_visual_property("node fill color", "fate", "d", FATES, FATE_COLORS, network=suid),
                        p4c.map_visual_property("node size", "fate", "d", FATES, FATE_SIZES, network=suid),
                    ]
                    p4c.create_visual_style(STYLE_NAME, defaults, mappings)
                    style_ready = True
                p4c.set_visual_style(STYLE_NAME, network=suid)
                p4c.fit_content(network=suid)
            except Exception as e:
                print("  style step skipped:", type(e).__name__, e)
            try:
                png = os.path.abspath(os.path.join(OUT_DIR, f"cytoscape_split_adaptive_{s}.png"))
                p4c.export_image(png, type="PNG", network=suid, overwrite_file=True)
            except Exception as e:
                print("  image export skipped:", type(e).__name__, e)
        if made:
            print(f"\nCytoscape: the new networks are in the Network panel (left side) under '{COLLECTION}'.")

if "ipykernel" in sys.modules:
    plt.show()
