"""
random_fractions.py
Random failure read off at fixed removal levels (10, 25, 35, 50, 65, 80, 90 %).

For each level it reports what is left of the network over many random runs, and
tests each targeted attack against the random runs AT THE SAME LEVEL.

Outputs (folder OUT_DIR):
  random_by_fraction.tsv        random failure at each level (mean, SD, range)
  targeted_by_fraction.tsv      each targeted attack vs random at each level (z, p)
  random_fraction_runs.tsv      every run at every level
  random_by_fraction.png        figure

Run the WHOLE file:   %run random_fractions.py   (notebook)   or   python random_fractions.py
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
OUT_DIR = "attack_random"
FRACTIONS = [0.10, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90]
N_RANDOM = 1000
SEED = 42                       # run 1, 2, 3 ... are the same runs as in random_failure.py
TARGETED_FILES = ["attack_adaptive/adaptive_curves.tsv", "attack_static/attack_curves.tsv"]
COLORS = {"degree": "#D55E00", "betweenness": "#0072B2", "closeness": "#009E73"}

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
STEPS = [round(f * N) for f in FRACTIONS]


def measure(keep):
    """Damage measures for the proteins still present (keep = True/False per protein)."""
    ids = np.flatnonzero(keep)
    sub = A[ids][:, ids]
    _, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    eff = 0.0
    if sub.nnz:
        D = shortest_path(sub, method="D", unweighted=True)
        ok = np.isfinite(D) & (D > 0)
        eff = (1.0 / D[ok]).sum() / (N * (N - 1))
    return int(sizes.max()), eff, int((sizes > 1).sum()), int((sizes == 1).sum())


LCC0, EFF0, COMP0, ISO0 = measure(np.ones(N, dtype=bool))
print(f"Intact network: {N} proteins, largest component {LCC0}, {COMP0} connected groups")

# ----------------------------------------------------------------------------
# 2. RANDOM RUNS, MEASURED AT EACH LEVEL
# ----------------------------------------------------------------------------
print(f"Running {N_RANDOM} random runs, measured at {len(FRACTIONS)} removal levels ...")
rows = []
for r in range(N_RANDOM):
    order = [IDX[n] for n in rng.permutation(nodes)]
    keep = np.ones(N, dtype=bool)
    done = 0
    for f, k in zip(FRACTIONS, STEPS):
        keep[order[done:k]] = False
        done = k
        lcc, eff, comps, iso = measure(keep)
        rows.append({"run": r + 1, "fraction_removed": f, "proteins_removed": k,
                     "lcc_n": lcc,
                     "lcc_percent_of_intact": 100 * lcc / LCC0,
                     "lcc_percent_of_survivors": 100 * lcc / (N - k),
                     "efficiency_percent_of_intact": 100 * eff / EFF0,
                     "components": comps, "isolated": iso})
    if (r + 1) % 200 == 0:
        print(f"  {r + 1}/{N_RANDOM} runs done")
runs = pd.DataFrame(rows)
runs.to_csv(f"{OUT_DIR}/random_fraction_runs.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------
# 3. RANDOM FAILURE AT EACH LEVEL
# ----------------------------------------------------------------------------
g = runs.groupby("fraction_removed")
table = pd.DataFrame({
    "proteins_removed": g["proteins_removed"].first(),
    "lcc_mean_%": g["lcc_percent_of_intact"].mean(),
    "lcc_sd": g["lcc_percent_of_intact"].std(),
    "lcc_p05_%": g["lcc_percent_of_intact"].quantile(0.05),
    "lcc_p95_%": g["lcc_percent_of_intact"].quantile(0.95),
    "expected_if_no_fragmentation_%": [100 * (1 - f) for f in FRACTIONS],
    "survivors_in_lcc_%": g["lcc_percent_of_survivors"].mean(),
    "efficiency_mean_%": g["efficiency_percent_of_intact"].mean(),
    "efficiency_sd": g["efficiency_percent_of_intact"].std(),
    "components_mean": g["components"].mean(),
    "isolated_mean": g["isolated"].mean(),
})
# damage beyond the proteins that were simply taken out
table["extra_loss_from_fragmentation_%"] = table["expected_if_no_fragmentation_%"] - table["lcc_mean_%"]
table.to_csv(f"{OUT_DIR}/random_by_fraction.tsv", sep="\t")

pd.set_option("display.width", 220)
print(f"\n=== RANDOM FAILURE at fixed removal levels ({N_RANDOM} runs) ===")
print("lcc = largest component, as % of the intact largest component")
print(table.round(2).to_string())

# ----------------------------------------------------------------------------
# 4. TARGETED ATTACKS AT THE SAME LEVELS
# ----------------------------------------------------------------------------
targeted = None
for path in TARGETED_FILES:
    if os.path.exists(path):
        tc = pd.read_csv(path, sep="\t")
        if "mode" not in tc.columns:
            tc["mode"] = "static"
        tc["attack"] = tc["mode"] + " " + tc["strategy"]
        out = []
        n_tests = len(FRACTIONS)
        for name, d in tc.groupby("attack", sort=False):
            d = d.set_index("step")
            for f, k in zip(FRACTIONS, STEPS):
                rnd = runs[runs["fraction_removed"] == f]
                for label, value, col in [
                        ("largest component", 100 * d.loc[k, "lcc_n"] / LCC0, "lcc_percent_of_intact"),
                        ("efficiency", 100 * d.loc[k, "efficiency"] / d.loc[0, "efficiency"],
                         "efficiency_percent_of_intact")]:
                    mu, sd = rnd[col].mean(), rnd[col].std()
                    p = ((rnd[col] <= value).sum() + 1) / (N_RANDOM + 1)
                    out.append({"attack": name, "fraction_removed": f, "measure": label,
                                "attack_%": value, "random_mean_%": mu, "difference": value - mu,
                                "z": (value - mu) / sd if sd > 0 else np.nan,
                                "p": p, "p_bonferroni": min(1.0, p * n_tests)})
        targeted = pd.DataFrame(out)
        targeted.to_csv(f"{OUT_DIR}/targeted_by_fraction.tsv", sep="\t", index=False)
        print(f"\nTargeted attacks read from {path}")
        for label in ["largest component", "efficiency"]:
            sub = targeted[targeted["measure"] == label]
            print(f"\n=== {label.upper()} remaining (% of intact): attack value, with p vs random in brackets ===")
            val = sub.pivot(index="attack", columns="fraction_removed", values="attack_%").loc[sub["attack"].unique()]
            pv = sub.pivot(index="attack", columns="fraction_removed", values="p_bonferroni").loc[val.index]
            show = val.round(1).astype(str) + " (" + pv.round(3).astype(str) + ")"
            show.loc["random (mean)"] = [f"{m:.1f}" for m in
                                         (table["lcc_mean_%"] if label == "largest component"
                                          else table["efficiency_mean_%"])]
            show.columns = [f"{int(round(100 * c))}%" for c in show.columns]
            print(show.to_string())
        print(f"\np = share of random runs at least as damaging as the attack, Bonferroni-corrected for "
              f"{n_tests} levels.\nSmallest possible corrected p: {min(1.0, n_tests / (N_RANDOM + 1)):.4f}")
        break
if targeted is None:
    print("\nNo targeted-attack output found; run static_attack.py / adaptive_attack.py to add the comparison.")

# ----------------------------------------------------------------------------
# 5. FIGURE
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
xpos = np.arange(len(FRACTIONS))
labels = [f"{int(round(100 * f))}%" for f in FRACTIONS]
for ax, col, measure_name, ylabel in [
        (axes[0], "lcc_percent_of_intact", "largest component", "Largest component (% of intact)"),
        (axes[1], "efficiency_percent_of_intact", "efficiency", "Global efficiency (% of intact)")]:
    data = [runs.loc[runs["fraction_removed"] == f, col].values for f in FRACTIONS]
    ax.boxplot(data, positions=xpos, widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="0.85", edgecolor="0.3"), medianprops=dict(color="black"),
               whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"), whis=(5, 95))
    if col == "lcc_percent_of_intact":
        ax.plot(xpos, [100 * (1 - f) for f in FRACTIONS], color="0.4", ls="--", lw=1,
                label="if only the removed proteins were lost")
    if targeted is not None:
        sub = targeted[targeted["measure"] == measure_name]
        for name in sub["attack"].unique():
            mode, strat = name.split(" ")
            d = sub[sub["attack"] == name]
            ax.plot(xpos, d["attack_%"].values, color=COLORS.get(strat, "red"),
                    marker="o" if mode == "adaptive" else "^", ms=6,
                    lw=1.6 if mode == "adaptive" else 1.0, ls="-" if mode == "adaptive" else ":", label=name)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Proteins removed")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
fig.suptitle(f"What is left at fixed removal levels: random failure ({N_RANDOM} runs; box = middle 50%, "
             "whiskers = 5-95%) and targeted attacks")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/random_by_fraction.png", dpi=200)
print(f"\nTables and figure saved in '{OUT_DIR}'.")

if "ipykernel" in sys.modules:
    plt.show()
