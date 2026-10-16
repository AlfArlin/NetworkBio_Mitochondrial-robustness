"""
results_table.py
LaTeX table of the attack results: static and adaptive removal by degree, betweenness
and closeness, and random failure, with every summary statistic. Reads only the saved
result tables; does not change any other script.

Needs (made by the earlier scripts):
  attack_static/attack_summary.tsv, attack_adaptive/adaptive_summary.tsv,
  attack_random/targeted_vs_random.tsv, attack_random/random_runs.tsv, mito_genes.tsv
Run:   %run results_table.py
Saves tables/results_table.tex   ->  in Overleaf: \\input{results_table}
Preamble needs:  \\usepackage{booktabs}  (and \\usepackage{graphicx} if FIT_TO_WIDTH)
"""
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
OUT = "tables/results_table.tex"
STRATEGIES = ["betweenness", "closeness", "degree"]      # row order inside each block
DIGITS = 3                     # decimals for R, R_E, f_1/2
BONFERRONI = True              # True: p values multiplied by the number of targeted attacks (6), capped at 1
SHOW_GAIN = True               # adaptive rows: % lower R_LCC than the static attack
FIT_TO_WIDTH = True            # \resizebox{\textwidth}: keeps the wide table inside the margins
LABEL = "tab:results"
CAPTION = "..."                # write your caption here (or in Overleaf)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
SYM = {}
if os.path.exists("mito_genes.tsv"):
    g = pd.read_csv("mito_genes.tsv", sep="\t").astype(str)
    SYM = dict(zip(g["orf"], g["symbol"]))

static = pd.read_csv("attack_static/attack_summary.tsv", sep="\t", index_col=0)
adaptive = pd.read_csv("attack_adaptive/adaptive_summary.tsv", sep="\t", index_col=0)
vs_random = pd.read_csv("attack_random/targeted_vs_random.tsv", sep="\t", index_col=0)
runs = pd.read_csv("attack_random/random_runs.tsv", sep="\t")
B = len(runs)                                         # number of random runs
n_tests = 6
p_floor = 1 / (B + 1)                                 # smallest p an empirical test with B runs can give


def num(x, d=DIGITS):
    if pd.isna(x):
        return "--"
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"     # proper minus sign in LaTeX


def pval(p):
    if pd.isna(p):
        return "--"
    if BONFERRONI:
        p = min(1.0, p * n_tests)
        floor = min(1.0, p_floor * n_tests)
    else:
        floor = p_floor
    if p <= floor * 1.0001:
        return f"$\\le${floor:.3f}"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def protein(x):
    return SYM.get(str(x), str(x))


rows = []
for mode, table in [("Static", static), ("Adaptive", adaptive)]:
    for i, s in enumerate(STRATEGIES):
        key = s if mode == "Static" else f"adaptive {s}"
        r = table.loc[key]
        t = vs_random.loc[f"{mode.lower()} {s}"]
        gain = ""
        if SHOW_GAIN:
            if mode == "Adaptive":
                st = static.loc[s, "R_lcc"]
                gain = f"{100 * (st - r['R_lcc']) / st:.1f}"
            else:
                gain = "--"
        rows.append([mode if i == 0 else "", s.capitalize(),
                     num(r["R_lcc"]), num(r["R_efficiency"]), num(r["f_half_lcc"]),
                     f"{int(r['largest_single_drop'])}", f"{protein(r['caused_by'])} ({int(r['at_step'])})",
                     num(t["R_lcc_z"], 2), pval(t["R_lcc_p"]), num(t["R_efficiency_z"], 2), pval(t["R_efficiency_p"])]
                    + ([gain] if SHOW_GAIN else []))

# random failure: mean +- SD over the B runs
def msd(col, d=DIGITS):
    return f"{runs[col].mean():.{d}f} $\\pm$ {runs[col].std():.{d}f}"

rows.append([f"Random ($n={B}$)", "--", msd("R_lcc"), msd("R_efficiency"), msd("f_half_lcc"),
             f"{runs['largest_single_drop'].mean():.0f} $\\pm$ {runs['largest_single_drop'].std():.0f}",
             "--", "--", "--", "--", "--"] + (["--"] if SHOW_GAIN else []))

head = ["Removal", "Centrality", "$R_{\\mathrm{LCC}}$", "$R_E$", "$f_{1/2}$", "$\\Delta_{\\max}$",
        "Caused by (step)", "$z_{R}$", "$p_{R}$", "$z_{R_E}$", "$p_{R_E}$"] + (["Gain (\\%)"] if SHOW_GAIN else [])
align = "ll" + "c" * (len(head) - 2)

lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{CAPTION}}}", f"\\label{{{LABEL}}}"]
if FIT_TO_WIDTH:
    lines.append("\\resizebox{\\textwidth}{!}{%")
lines += [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
for i, r in enumerate(rows):
    if i in (3, 6):
        lines.append("\\midrule")
    lines.append(" & ".join(r) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}" + ("}" if FIT_TO_WIDTH else "")]
note = (f"$R_{{\\mathrm{{LCC}}}}$, $R_E$: robustness indices of the largest connected component and of global "
        f"efficiency; $f_{{1/2}}$: fraction removed when the largest component first falls to half its size; "
        f"$\\Delta_{{\\max}}$: largest single-step loss of proteins from the largest component; "
        f"$z$, $p$: z score and one-sided empirical p value against {B} random-failure runs"
        + (f", Bonferroni-corrected for {n_tests} attacks" if BONFERRONI else "")
        + "; random row: mean $\\pm$ SD"
        + ("; Gain: reduction of $R_{\\mathrm{LCC}}$ of the adaptive relative to the static attack." if SHOW_GAIN else "."))
lines += ["\\par\\smallskip", f"\\footnotesize{{{note}}}", "\\end{table}"]

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved {OUT}")
