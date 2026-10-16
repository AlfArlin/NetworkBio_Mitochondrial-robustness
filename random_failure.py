"""
random_failure.py
Random failure of the yeast mitochondrial PPI network (the baseline for all attacks).

Random failure = proteins are removed in a uniformly random order, one at a time.
There is no ranking, so it is repeated many times and described as a distribution.

Degree, betweenness and closeness enter in three ways:
  1. the targeted attacks (read from the earlier scripts' output) are placed on the
     random distribution, giving a z score and an empirical p value for each;
  2. per run: does a random run do more damage when it happens to hit proteins with
     high degree / betweenness / closeness early?
  3. per protein: average damage caused when that protein is removed at a random
     moment, compared with its degree / betweenness / closeness.

Outputs : tables + figures in OUT_DIR, and views in Cytoscape (if it is open)
Run the WHOLE file:   %run random_failure.py   (notebook)   or   python random_failure.py
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path, connected_components

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
OUT_DIR = "attack_random"
N_RANDOM = 500                 # number of random runs (the first 100 equal the earlier baseline)
SEED = 42
EARLY_FRACTION = 0.10          # "early" removals = first 10% of proteins
SNAPSHOT_FRACTION = 0.10       # network pictures are taken after removing this fraction
CENTRALITIES = ["degree", "betweenness", "closeness"]
COLORS = {"degree": "#D55E00", "betweenness": "#0072B2", "closeness": "#009E73"}
TARGETED_FILES = ["attack_adaptive/adaptive_curves.tsv", "attack_static/attack_curves.tsv"]
USE_CYTOSCAPE = True
BASE_NETWORK_TITLE = "Yeast mitochondrial PPI"
COLLECTION = "Random failure views"
STYLE_NAME = "Random failure"

STATUSES = ["largest component", "smaller fragment", "isolated"]
STATUS_COLORS = ["#0072B2", "#E69F00", "#BBBBBB"]

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(G.nodes())
N = len(nodes)
IDX = {n: i for i, n in enumerate(nodes)}
A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr", dtype=float)

SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))


def sym(n):
    return SYM.get(n, n)


cent = pd.DataFrame({"degree": dict(G.degree()),
                     "betweenness": nx.betweenness_centrality(G),
                     "closeness": nx.closeness_centrality(G)}).loc[nodes]

# ----------------------------------------------------------------------------
# 2. ONE REMOVAL RUN (same damage measures as the attack scripts)
# ----------------------------------------------------------------------------
def run(order_idx):
    """Remove proteins in the given order; return the damage after every step."""
    keep = np.ones(N, dtype=bool)
    lcc = np.zeros(N + 1, dtype=int)
    comps = np.zeros(N + 1, dtype=int)
    eff = np.zeros(N + 1)
    for step in range(N + 1):
        if step > 0:
            keep[order_idx[step - 1]] = False
        ids = np.flatnonzero(keep)
        if len(ids) == 0:
            break
        sub = A[ids][:, ids]
        _, lab = connected_components(sub, directed=False)
        sizes = np.bincount(lab)
        lcc[step] = sizes.max()
        comps[step] = (sizes > 1).sum()
        if sub.nnz:
            D = shortest_path(sub, method="D", unweighted=True)
            ok = np.isfinite(D) & (D > 0)
            eff[step] = (1.0 / D[ok]).sum() / (N * (N - 1))
    return lcc, comps, eff


def first_at_or_below(values, threshold):
    hit = np.flatnonzero(values <= threshold)
    return hit[0] / N if len(hit) else np.nan


# ----------------------------------------------------------------------------
# 3. RUN THE RANDOM FAILURES
# ----------------------------------------------------------------------------
print(f"Intact network: {N} proteins, {G.number_of_edges()} interactions")
print(f"Running {N_RANDOM} random-failure runs ...")
k_early = max(1, round(EARLY_FRACTION * N))
LCC = np.zeros((N_RANDOM, N + 1), dtype=int)
COMP = np.zeros((N_RANDOM, N + 1), dtype=int)
EFF = np.zeros((N_RANDOM, N + 1))
orders, run_rows = [], []
damage_sum, damage_max, worst_count = np.zeros(N), np.zeros(N), np.zeros(N, dtype=int)

for r in range(N_RANDOM):
    order = [IDX[n] for n in rng.permutation(nodes)]
    orders.append(order)
    lcc, comps, eff = run(order)
    LCC[r], COMP[r], EFF[r] = lcc, comps, eff
    drop = lcc[:-1] - lcc[1:]                       # proteins lost from largest component per step
    damage_sum[order] += drop
    damage_max[order] = np.maximum(damage_max[order], drop)
    worst = int(drop.argmax())
    worst_count[order[worst]] += 1
    early = order[:k_early]
    run_rows.append({
        "run": r + 1,
        "R_lcc": lcc[1:].mean() / N,
        "R_efficiency": (eff[1:] / eff[0]).mean(),
        "f_half_lcc": first_at_or_below(lcc, lcc[0] / 2),
        "f_half_efficiency": first_at_or_below(eff, eff[0] / 2),
        "largest_single_drop": int(drop[worst]),
        "caused_by": sym(nodes[order[worst]]),
        "at_step": worst + 1,
        **{f"early_mean_{c}": cent[c].values[early].mean() for c in CENTRALITIES},
    })
    if (r + 1) % 50 == 0:
        print(f"  {r + 1}/{N_RANDOM} runs done")
runs = pd.DataFrame(run_rows)

# ----------------------------------------------------------------------------
# 4. STATISTICS
# ----------------------------------------------------------------------------
metrics = ["R_lcc", "R_efficiency", "f_half_lcc", "f_half_efficiency", "largest_single_drop"]
dist = runs[metrics].agg(["mean", "std", "median", "min", "max"]).T
dist["p05"] = runs[metrics].quantile(0.05)
dist["p95"] = runs[metrics].quantile(0.95)
dist["cv_percent"] = 100 * dist["std"] / dist["mean"]
sem = dist["std"] / np.sqrt(N_RANDOM)
dist["ci95_low"], dist["ci95_high"] = dist["mean"] - 1.96 * sem, dist["mean"] + 1.96 * sem
dist = dist[["mean", "ci95_low", "ci95_high", "std", "cv_percent", "median", "p05", "p95", "min", "max"]]

# ---- 4a. targeted attacks placed on the random distribution
targeted, targeted_curves = None, None
for path in TARGETED_FILES:
    if os.path.exists(path):
        tc = pd.read_csv(path, sep="\t")
        if "mode" not in tc.columns:
            tc["mode"] = "static"
        tc["attack"] = tc["mode"] + " " + tc["strategy"]
        targeted_curves = tc
        rows = {}
        for name, d in tc.groupby("attack", sort=False):
            d = d.sort_values("step")
            lcc_t, eff_t = d["lcc"].values, d["efficiency"].values
            rows[name] = {"R_lcc": lcc_t[1:].mean(), "R_efficiency": (eff_t[1:] / eff_t[0]).mean(),
                          "f_half_lcc": first_at_or_below(lcc_t, lcc_t[0] / 2),
                          "f_half_efficiency": first_at_or_below(eff_t, eff_t[0] / 2)}
        targeted = pd.DataFrame(rows).T
        for col in ["R_lcc", "R_efficiency"]:
            mu, sd = runs[col].mean(), runs[col].std()
            targeted[f"{col}_z"] = (targeted[col] - mu) / sd
            targeted[f"{col}_p"] = [((runs[col] <= v).sum() + 1) / (N_RANDOM + 1) for v in targeted[col]]
            targeted[f"{col}_ratio_to_random"] = targeted[col] / mu
        print(f"Targeted attacks read from {path}")
        break
if targeted is None:
    print("No targeted-attack output found; run static_attack.py / adaptive_attack.py to add the comparison.")

# ---- 4b. per run: does hitting central proteins early make a random run worse?
run_corr = pd.DataFrame({
    "spearman_with_R_lcc": [runs[f"early_mean_{c}"].corr(runs["R_lcc"], method="spearman") for c in CENTRALITIES],
    "spearman_with_R_efficiency": [runs[f"early_mean_{c}"].corr(runs["R_efficiency"], method="spearman")
                                   for c in CENTRALITIES],
}, index=CENTRALITIES)

# ---- 4c. per protein: average damage when removed at a random moment
node_table = cent.reset_index().rename(columns={"index": "id"})
node_table["symbol"] = node_table["id"].map(sym)
node_table["mean_random_damage"] = damage_sum / N_RANDOM
node_table["max_random_damage"] = damage_max
node_table["times_most_damaging"] = worst_count
node_corr = pd.DataFrame({"spearman_with_mean_random_damage":
                          [node_table[c].corr(node_table["mean_random_damage"], method="spearman")
                           for c in CENTRALITIES]}, index=CENTRALITIES)

pd.set_option("display.width", 220)
print(f"\n=== RANDOM FAILURE: distribution over {N_RANDOM} runs ===")
print(dist.round(4).to_string())
if targeted is not None:
    print("\n=== Targeted attacks against the random distribution ===")
    print(targeted.round(4).to_string())
    print(f"(smallest possible p with {N_RANDOM} runs: {1 / (N_RANDOM + 1):.4f})")
print(f"\n=== Per run: mean centrality of the first {k_early} removed vs damage (negative = more damage) ===")
print(run_corr.round(3).to_string())
print("\n=== Per protein: centrality vs average damage under random failure ===")
print(node_corr.round(3).to_string())
print("\n=== 15 proteins with the highest average damage under random failure ===")
print(node_table.sort_values("mean_random_damage", ascending=False).head(15)
      [["symbol", "degree", "betweenness", "closeness", "mean_random_damage", "max_random_damage",
        "times_most_damaging"]].round(4).to_string(index=False))

runs.to_csv(f"{OUT_DIR}/random_runs.tsv", sep="\t", index=False)
dist.to_csv(f"{OUT_DIR}/random_distribution.tsv", sep="\t")
node_table.to_csv(f"{OUT_DIR}/node_random_table.tsv", sep="\t", index=False)
run_corr.join(node_corr).to_csv(f"{OUT_DIR}/centrality_vs_random_damage.tsv", sep="\t")
if targeted is not None:
    targeted.to_csv(f"{OUT_DIR}/targeted_vs_random.tsv", sep="\t")
x = np.arange(N + 1) / N
EREL = EFF / EFF[:, [0]]                      # efficiency relative to the intact network
pd.DataFrame({"fraction_removed": x,
              "lcc_mean": LCC.mean(0) / N, "lcc_p05": np.quantile(LCC, 0.05, 0) / N,
              "lcc_p95": np.quantile(LCC, 0.95, 0) / N,
              "lcc_min": LCC.min(0) / N, "lcc_max": LCC.max(0) / N,
              "efficiency_mean": EFF.mean(0),
              "eff_rel_mean": EREL.mean(0), "eff_rel_p05": np.quantile(EREL, 0.05, 0),
              "eff_rel_p95": np.quantile(EREL, 0.95, 0),
              "components_mean": COMP.mean(0), "components_p05": np.quantile(COMP, 0.05, 0),
              "components_p95": np.quantile(COMP, 0.95, 0)}
             ).to_csv(f"{OUT_DIR}/random_curves.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------
# 5. FIGURES
# ----------------------------------------------------------------------------
E0 = EFF[0, 0]
panels = [(LCC / N, "lcc", 1.0, "Largest component (fraction of all proteins)"),
          (EFF / E0, "efficiency", E0, "Global efficiency (relative to intact)"),
          (COMP.astype(float), "components", 1.0, "Connected groups of 2+ proteins")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (M, col, norm, label) in zip(axes, panels):
    ax.fill_between(x, M.min(0), M.max(0), color="0.85", label="random, all runs (min-max)")
    ax.fill_between(x, np.quantile(M, 0.05, 0), np.quantile(M, 0.95, 0), color="0.6", alpha=0.6,
                    label="random, 5-95% of runs")
    for r in range(min(20, N_RANDOM)):
        ax.plot(x, M[r], color="0.35", lw=0.3, alpha=0.5)
    ax.plot(x, M.mean(0), color="black", lw=2, label="random, mean")
    if targeted_curves is not None:
        for name, d in targeted_curves.groupby("attack", sort=False):
            mode, strat = name.split(" ")
            d = d.sort_values("step")
            ax.plot(d["fraction_removed"], d[col] / norm, color=COLORS.get(strat, "red"),
                    lw=1.6 if mode == "adaptive" else 1.0, ls="-" if mode == "adaptive" else ":", label=name)
    ax.set_xlabel("Fraction of proteins removed")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.suptitle(f"Random failure, {N_RANDOM} runs ({N} proteins, {G.number_of_edges()} interactions); "
             "thin grey lines are 20 individual runs")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/random_curves.png", dpi=200)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4.6))
for ax, col, title in zip(axes2[:2], ["R_lcc", "R_efficiency"],
                          ["Robustness index R (largest component)", "Robustness index (efficiency)"]):
    ax.hist(runs[col], bins=30, color="0.6", edgecolor="white", label=f"random runs (n={N_RANDOM})")
    ax.axvline(runs[col].mean(), color="black", lw=2, label="random, mean")
    if targeted is not None:
        for name, v in targeted[col].items():
            mode, strat = name.split(" ")
            ax.axvline(v, color=COLORS.get(strat, "red"), lw=1.8, ls="-" if mode == "adaptive" else ":", label=name)
    ax.set_xlabel(title)
    ax.set_ylabel("Number of runs")
    ax.legend(fontsize=7)
running = runs["R_lcc"].expanding().mean()
axes2[2].plot(np.arange(1, N_RANDOM + 1), running, color="black")
axes2[2].axhline(runs["R_lcc"].mean(), color="0.5", ls="--")
axes2[2].set_xlabel("Number of random runs included")
axes2[2].set_ylabel("Running mean of R (largest component)")
axes2[2].set_title("Has the baseline settled?", fontsize=10)
axes2[2].grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/random_distribution.png", dpi=200)

fig3, axes3 = plt.subplots(2, 3, figsize=(17, 9))
for j, c in enumerate(CENTRALITIES):
    ax = axes3[0, j]
    ax.scatter(runs[f"early_mean_{c}"], runs["R_lcc"], s=12, color=COLORS[c], alpha=0.6)
    ax.set_xlabel(f"Mean {c} of the first {k_early} proteins removed")
    ax.set_ylabel("R of that run (lower = more damage)")
    ax.set_title(f"Per run: Spearman = {run_corr.loc[c, 'spearman_with_R_lcc']:.2f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax = axes3[1, j]
    ax.scatter(node_table[c], node_table["mean_random_damage"], s=12, color=COLORS[c], alpha=0.6)
    for _, row in node_table.nlargest(5, "mean_random_damage").iterrows():
        ax.annotate(row["symbol"], (row[c], row["mean_random_damage"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"{c} on the intact network")
    ax.set_ylabel("Average proteins lost when removed at random")
    ax.set_title(f"Per protein: Spearman = {node_corr.loc[c, 'spearman_with_mean_random_damage']:.2f}", fontsize=10)
    ax.grid(alpha=0.3)
fig3.suptitle("What makes random failure hurt: centrality of the proteins that happen to be hit")
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/centrality_vs_random_damage.png", dpi=200)

# ----------------------------------------------------------------------------
# 6. NETWORK PICTURES: three random runs after SNAPSHOT_FRACTION of removals
# ----------------------------------------------------------------------------
def make_layout(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.kamada_kawai_layout(G.subgraph(comps[0]))
    per_row, dx, dy = 12, 2.0 / 11, 0.16
    for i, c in enumerate(comps[1:]):
        cx, cy = -1 + (i % per_row) * dx, -1.25 - (i // per_row) * dy
        for n, (px, py) in nx.circular_layout(G.subgraph(c), scale=0.035 + 0.008 * len(c)).items():
            pos[n] = (cx + px, cy + py)
    return pos


k_snap = max(1, round(SNAPSHOT_FRACTION * N))
ranked = runs.sort_values("R_lcc").reset_index(drop=True)
chosen = {"most damaging run": int(ranked.iloc[0]["run"]),
          "typical run (median R)": int(ranked.iloc[len(ranked) // 2]["run"]),
          "least damaging run": int(ranked.iloc[-1]["run"])}

snapshots = {}
for label, run_no in chosen.items():
    removed = [nodes[i] for i in orders[run_no - 1][:k_snap]]
    H = G.copy()
    H.remove_nodes_from(removed)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    status = {}
    for i, c in enumerate(comps):
        for n in c:
            status[n] = "largest component" if i == 0 else ("isolated" if len(c) == 1 else "smaller fragment")
    table = pd.DataFrame({"id": list(H.nodes())})
    table["symbol"] = table["id"].map(sym)
    table["status"] = table["id"].map(status)
    snapshots[label] = dict(run=run_no, removed=removed, H=H, status=status, table=table,
                            lcc=len(comps[0]), R=float(runs.loc[run_no - 1, "R_lcc"]))
    print(f"\n=== {label} (run {run_no}, R = {snapshots[label]['R']:.3f}) after {k_snap} removals ===")
    print(f"Largest component: {LCC[run_no - 1, 0]} -> {len(comps[0])} proteins")
    print(f"Removed: {', '.join(sym(n) for n in removed)}")

pos = make_layout(G)
color_of = dict(zip(STATUSES, STATUS_COLORS))
fig4, axes4 = plt.subplots(1, 3, figsize=(24, 9))
for ax, (label, s) in zip(axes4, snapshots.items()):
    nx.draw_networkx_edges(s["H"], pos, ax=ax, edge_color="0.7", width=0.4, alpha=0.6)
    for st in STATUSES[::-1]:
        nl = [n for n in s["H"] if s["status"][n] == st]
        nx.draw_networkx_nodes(s["H"], pos, nodelist=nl, node_color=color_of[st], node_size=24, ax=ax,
                               linewidths=0, label=f"{st} ({len(nl)})")
    nx.draw_networkx_nodes(G, pos, nodelist=s["removed"], node_color="black", node_size=24, linewidths=0,
                           ax=ax, label=f"removed ({len(s['removed'])})")      # black dot, as in the targeted views
    ax.set_title(f"{label}: run {s['run']}, R = {s['R']:.3f}\nlargest component {s['lcc']} proteins")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_axis_off()
fig4.suptitle(f"Random failure after removing {k_snap} proteins ({100 * SNAPSHOT_FRACTION:.0f}%)", fontsize=14)
fig4.tight_layout()
fig4.savefig(f"{OUT_DIR}/random_snapshots.png", dpi=200)
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
            try:
                cols = node_table[["id", "mean_random_damage", "max_random_damage", "times_most_damaging"]]
                p4c.load_table_data(cols, data_key_column="id", table_key_column="name", network=base)
                print(f"Cytoscape: random-damage columns added to '{BASE_NETWORK_TITLE}'")
            except Exception as e:
                print("Cytoscape: could not add columns to the original network:", type(e).__name__, e)

        style_ready, made = False, []
        for label, s in snapshots.items():
            title = f"Random: {label} (run {s['run']}, {k_snap} removed)"
            try:
                if title in p4c.get_network_list():
                    p4c.delete_network(title)
                e_df = pd.DataFrame([(str(a), str(b)) for a, b in s["H"].edges()], columns=["source", "target"])
                e_df["interaction"] = "pp"
                suid = p4c.create_network_from_data_frames(nodes=s["table"].copy(), edges=e_df,
                                                           title=title, collection=COLLECTION)
                made.append(title)
                print(f"Cytoscape: created '{title}'")
            except Exception as e:
                print(f"Cytoscape: could not create '{label}':", type(e).__name__, e)
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
                        p4c.map_visual_property("node fill color", "status", "d", STATUSES, STATUS_COLORS,
                                                network=suid),
                    ]
                    p4c.create_visual_style(STYLE_NAME, defaults, mappings)
                    style_ready = True
                p4c.set_visual_style(STYLE_NAME, network=suid)
                p4c.fit_content(network=suid)
            except Exception as e:
                print("  style step skipped:", type(e).__name__, e)
            try:
                png = os.path.abspath(os.path.join(OUT_DIR, f"cytoscape_random_run_{s['run']}.png"))
                p4c.export_image(png, type="PNG", network=suid, overwrite_file=True)
            except Exception as e:
                print("  image export skipped:", type(e).__name__, e)
        if made:
            print(f"\nCytoscape: the new networks are in the Network panel (left side) under '{COLLECTION}'.")

if "ipykernel" in sys.modules:
    plt.show()
