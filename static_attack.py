"""
static_attack.py
Static targeted attacks on the yeast mitochondrial PPI network.

Strategies : degree, betweenness, closeness (ranked ONCE on the intact network)
Baseline   : random removal, repeated N_RANDOM times
Damage     : largest connected component, global efficiency, fragmentation
Outputs    : tables + figures in OUT_DIR, and views in Cytoscape (if it is open)

Run in a notebook cell :  %run static_attack.py
Run in a terminal      :  python static_attack.py
Needs: pip install pandas numpy networkx scipy matplotlib py4cytoscape
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
EDGES_FILE = "mito_ppi_edges.tsv"      # columns: source, target (ORF ids)
GENES_FILE = "mito_genes.tsv"          # columns: orf, symbol (optional file)
OUT_DIR = "attack_static"
N_RANDOM = 100                          # random-failure repetitions
SEED = 42
SNAPSHOT_FRACTION = 0.10                # Cytoscape snapshot: top 10% removed
USE_CYTOSCAPE = True
BASE_NETWORK_TITLE = "Yeast mitochondrial PPI"
STRATEGIES = ["degree", "betweenness", "closeness"]
COLORS = {"degree": "#D55E00", "betweenness": "#0072B2", "closeness": "#009E73"}

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. LOAD THE NETWORK
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
N = G.number_of_nodes()

if os.path.exists(GENES_FILE):
    g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(g["orf"], g["symbol"]))
else:
    SYM = {}
sym = lambda n: SYM.get(n, n)


# ----------------------------------------------------------------------------
# 2. DAMAGE MEASURES
# ----------------------------------------------------------------------------
def efficiency(H, n_original):
    """Global efficiency, normalised by the ORIGINAL number of node pairs."""
    if H.number_of_edges() == 0:
        return 0.0
    A = nx.to_scipy_sparse_array(H, weight=None, format="csr", dtype=float)
    D = shortest_path(A, method="D", unweighted=True)
    ok = np.isfinite(D) & (D > 0)
    return float((1.0 / D[ok]).sum() / (n_original * (n_original - 1)))


def measure(H, n_original):
    sizes = [len(c) for c in nx.connected_components(H)]
    return {
        "lcc_n": max(sizes, default=0),                    # proteins in largest component
        "lcc": max(sizes, default=0) / n_original,         # same, as fraction of all proteins
        "components": sum(1 for s in sizes if s > 1),      # connected groups of 2+ proteins
        "isolated": sum(1 for s in sizes if s == 1),       # proteins left with no partner
        "edges": H.number_of_edges(),
        "efficiency": efficiency(H, n_original),
    }


def run_attack(G, order):
    """Remove nodes in the given fixed order; measure the network after each step."""
    H, n0 = G.copy(), G.number_of_nodes()
    rows = [{"step": 0, "removed": None, **measure(H, n0)}]
    for i, node in enumerate(order, start=1):
        H.remove_node(node)
        rows.append({"step": i, "removed": node, **measure(H, n0)})
    df = pd.DataFrame(rows)
    df["fraction_removed"] = df["step"] / n0
    df["lcc_drop"] = -df["lcc_n"].diff().fillna(0).astype(int)   # proteins lost from LCC at this step
    return df


def summarise(df):
    S0, E0 = df["lcc"].iloc[0], df["efficiency"].iloc[0]
    worst = df.loc[df["lcc_drop"].idxmax()]
    half_lcc = df.loc[df["lcc"] <= S0 / 2, "fraction_removed"]
    half_eff = df.loc[df["efficiency"] <= E0 / 2, "fraction_removed"]
    return {
        "R_lcc": df["lcc"].iloc[1:].mean(),                       # Schneider robustness index
        "R_efficiency": (df["efficiency"].iloc[1:] / E0).mean(),  # same idea, on efficiency
        "f_half_lcc": half_lcc.min(),          # fraction removed to halve the LCC
        "f_half_efficiency": half_eff.min(),   # fraction removed to halve efficiency
        "largest_single_drop": int(worst["lcc_drop"]),
        "caused_by": sym(worst["removed"]),
        "at_step": int(worst["step"]),
    }


# ----------------------------------------------------------------------------
# 3. STATIC RANKINGS (computed once, on the intact network)
# ----------------------------------------------------------------------------
scores = pd.DataFrame({
    "degree": dict(G.degree()),
    "betweenness": nx.betweenness_centrality(G),
    "closeness": nx.closeness_centrality(G),
}).rename_axis("id").reset_index()

orders = {}
for s in STRATEGIES:
    # highest score first; ties broken by ORF name so the result is reproducible
    ranked = scores.sort_values([s, "id"], ascending=[False, True])["id"].tolist()
    orders[s] = ranked
    scores[f"rank_{s}"] = scores["id"].map({n: i + 1 for i, n in enumerate(ranked)})

# ----------------------------------------------------------------------------
# 4. RUN THE ATTACKS
# ----------------------------------------------------------------------------
print(f"Intact network: {N} proteins, {G.number_of_edges()} interactions")
results = {}
for s in STRATEGIES:
    print(f"  running static {s} attack ...")
    results[s] = run_attack(G, orders[s])

print(f"  running {N_RANDOM} random-failure runs ...")
nodes = list(G.nodes())
rand_runs = [run_attack(G, list(rng.permutation(nodes))) for _ in range(N_RANDOM)]
rand = {m: pd.concat([r[m] for r in rand_runs], axis=1) for m in ["lcc", "efficiency", "components"]}
rand_summ = pd.DataFrame([summarise(r) for r in rand_runs])

# ----------------------------------------------------------------------------
# 5. STATISTICS
# ----------------------------------------------------------------------------
summary = pd.DataFrame({s: summarise(results[s]) for s in STRATEGIES}).T
for col in ["R_lcc", "R_efficiency"]:
    mu, sd = rand_summ[col].mean(), rand_summ[col].std()
    summary[f"{col}_z_vs_random"] = (summary[col].astype(float) - mu) / sd
    # share of random runs that were at least as damaging as the attack
    summary[f"{col}_p_vs_random"] = [
        ((rand_summ[col] <= v).sum() + 1) / (N_RANDOM + 1) for v in summary[col].astype(float)]
summary.loc["random (mean)"] = {
    "R_lcc": rand_summ["R_lcc"].mean(), "R_efficiency": rand_summ["R_efficiency"].mean(),
    "f_half_lcc": rand_summ["f_half_lcc"].mean(),
    "f_half_efficiency": rand_summ["f_half_efficiency"].mean()}

k = max(1, round(SNAPSHOT_FRACTION * N))
rank_corr = scores[[f"rank_{s}" for s in STRATEGIES]].corr(method="spearman")
overlap = pd.DataFrame(
    [[len(set(orders[a][:k]) & set(orders[b][:k])) for b in STRATEGIES] for a in STRATEGIES],
    index=STRATEGIES, columns=STRATEGIES)

pd.set_option("display.width", 200)
print("\n=== SUMMARY (lower R = more damaging attack) ===")
print(summary.round(4).to_string())
print("\n=== Agreement between the three rankings (Spearman) ===")
print(rank_corr.round(3).to_string())
print(f"\n=== Shared proteins among the top {k} targets ===")
print(overlap.to_string())
for s in STRATEGIES:
    top = results[s].iloc[1:16]
    print(f"\n=== First 15 targets, {s} attack (proteins lost from LCC at that step) ===")
    print(", ".join(f"{sym(n)} ({d})" for n, d in zip(top["removed"], top["lcc_drop"])))

# ---- save tables
curves = pd.concat([df.assign(strategy=s) for s, df in results.items()], ignore_index=True)
curves["symbol"] = curves["removed"].map(lambda n: sym(n) if n else None)
curves.to_csv(f"{OUT_DIR}/attack_curves.tsv", sep="\t", index=False)
summary.to_csv(f"{OUT_DIR}/attack_summary.tsv", sep="\t")
rand_summ.to_csv(f"{OUT_DIR}/random_runs_summary.tsv", sep="\t", index=False)

node_table = scores.copy()
node_table["symbol"] = node_table["id"].map(sym)
for s in STRATEGIES:
    node_table[f"damage_{s}"] = node_table["id"].map(
        dict(zip(results[s]["removed"].iloc[1:], results[s]["lcc_drop"].iloc[1:])))
    node_table[f"top_target_{s}"] = node_table[f"rank_{s}"] <= k
node_table.to_csv(f"{OUT_DIR}/node_attack_table.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------
# 6. FIGURES
# ----------------------------------------------------------------------------
x = results[STRATEGIES[0]]["fraction_removed"]
E0 = results[STRATEGIES[0]]["efficiency"].iloc[0]
panels = [("lcc", "Largest component (fraction of all proteins)", 1.0),
          ("efficiency", "Global efficiency (relative to intact)", E0),
          ("components", "Connected groups of 2+ proteins", 1.0)]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, (m, label, norm) in zip(axes, panels):
    r = rand[m] / norm
    ax.fill_between(x, r.quantile(0.05, axis=1), r.quantile(0.95, axis=1),
                    color="0.6", alpha=0.35, label="random, 5-95% of runs")
    ax.plot(x, r.mean(axis=1), color="0.2", lw=1.5, ls="--", label="random, mean")
    for s in STRATEGIES:
        ax.plot(x, results[s][m] / norm, color=COLORS[s], lw=2, label=f"{s} attack")
    ax.set_xlabel("Fraction of proteins removed")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"Static targeted attacks ({N} proteins, {G.number_of_edges()} interactions)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/attack_curves.png", dpi=200)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in zip(axes2, ["R_lcc", "R_efficiency"],
                          ["Robustness index R (largest component)", "Robustness index (efficiency)"]):
    vals = summary.loc[STRATEGIES, col].astype(float)
    ax.bar(STRATEGIES, vals, color=[COLORS[s] for s in STRATEGIES])
    mu, sd = rand_summ[col].mean(), rand_summ[col].std()
    ax.axhline(mu, color="0.2", ls="--", label="random, mean")
    ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="0.6", alpha=0.3, label="random, mean ± 2 SD")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/robustness_index.png", dpi=200)

# ----------------------------------------------------------------------------
# 7. CYTOSCAPE
# ----------------------------------------------------------------------------
if USE_CYTOSCAPE:
    try:
        import py4cytoscape as p4c
        p4c.cytoscape_ping()

        if BASE_NETWORK_TITLE in p4c.get_network_list():
            base = p4c.get_network_suid(BASE_NETWORK_TITLE)
        else:
            base = p4c.create_network_from_data_frames(
                nodes=node_table[["id", "symbol"]],
                edges=edges.assign(interaction="pp"),
                title=BASE_NETWORK_TITLE, collection="SGD GO:0005739")
            p4c.layout_network("force-directed", network=base)

        # attack columns on every node: scores, ranks, damage, top-target flags
        p4c.load_table_data(node_table, data_key_column="id", table_key_column="name", network=base)

        # one snapshot per strategy: same layout, top targets deleted
        for s in STRATEGIES:
            title = f"After {s} attack (top {k} removed)"
            if title in p4c.get_network_list():
                p4c.delete_network(title)
            p4c.set_current_network(base)
            clone = p4c.clone_network(network=base)
            p4c.rename_network(title, network=clone)
            p4c.clear_selection(network=clone)
            p4c.select_nodes(orders[s][:k], by_col="name", network=clone)
            p4c.delete_selected_nodes(network=clone)

        # intact network: red = earliest betweenness targets, size = degree
        p4c.set_current_network(base)
        try:
            p4c.set_node_color_mapping("rank_betweenness", [1, k, N],
                                       ["#B2182B", "#FDDBC7", "#F0F0F0"], mapping_type="c")
            p4c.set_node_size_mapping("degree", [1, int(scores["degree"].max())], [25, 80])
        except Exception as e:
            print("Cytoscape styling skipped:", e)
        print(f"\nCytoscape: attack columns added to '{BASE_NETWORK_TITLE}', plus 3 snapshot networks.")
    except Exception as e:
        print("\nCytoscape step skipped (is Cytoscape open?):", type(e).__name__, e)
        print(f"You can import {OUT_DIR}/node_attack_table.tsv by hand: File > Import > Table from File.")

print(f"\nDone. Tables and figures are in the '{OUT_DIR}' folder.")
if "ipykernel" in sys.modules:
    plt.show()
