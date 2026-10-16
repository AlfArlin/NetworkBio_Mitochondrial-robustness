"""
style.py -- the ONE place for colours, fonts and sizes of every report figure.

Read by figures.py, network_layout.py and split_view.py (and through them, by the
Cytoscape styles), so changing a value here changes it everywhere.

Colour rules used throughout (one meaning per colour, in every figure):
  * blue / orange / aqua  = the three removal strategies (identity)
  * line style            = static (dashed) vs adaptive (solid); colour never encodes mode
  * neutral greys         = random failure, and network context
  * red / yellow          = damage in the network views (critical protein / part that splits off)
  * blue ramp             = betweenness magnitude on the overview network
The strategy trio was checked as a set (all pairs) for colour-blind separation.
"""
import matplotlib as mpl
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# 1. Removal strategies (identity)
# ---------------------------------------------------------------------------
STRATEGY_ORDER = ["betweenness", "degree", "closeness"]          # legend order
STRATEGY_COLORS = {"betweenness": "#2a78d6",                       # blue
                   "degree":      "#eb6834",                       # orange
                   "closeness":   "#1baf7a"}                       # aqua
MODE_ORDER = ["adaptive", "static"]
MODE_LINESTYLE = {"adaptive": "-", "static": (0, (4, 2))}          # solid / dashed
MODE_MARKER = {"adaptive": "o", "static": "^"}

# ---------------------------------------------------------------------------
# 2. Random failure (the reference) -- neutral, never a strategy colour
# ---------------------------------------------------------------------------
RANDOM_LINE = "#52514e"
RANDOM_BAND = "#d6d5d0"        # 5-95 % of runs
RANDOM_BAND_LIGHT = "#ecebe7"  # min-max of runs

# ---------------------------------------------------------------------------
# 3. Network views (split views, overview network, Cytoscape)
# ---------------------------------------------------------------------------
CRITICAL = "#e34948"           # red: the critical protein
SPLIT = "#eda100"              # yellow: proteins that split off
SPLIT_EDGE = "#c98500"         # darker yellow: interactions inside the split-off part
REMOVED = "#0b0b0b"            # black: removed earlier
STAYS = "#86b6ef"              # light blue: stays in the largest component (same blue as the plain network)
DISCONNECTED = "#e4e3df"       # light grey: already disconnected
EDGE = "#cfcec9"               # ordinary interactions
NODE_BORDER = "#0b0b0b"

# Plain network figure (the intact PPI network): one colour, equal circles, no labels
NETWORK_NODE = "#86b6ef"       # light blue
NETWORK_NODE_RIM = "#ffffff"   # thin white rim: keeps touching circles in dense complexes apart
NETWORK_NODE_RIM_PX = 1.5
NETWORK_EDGE = "#b9b8b2"       # light grey
NETWORK_EDGE_ALPHA = 0.45      # see-through edges, so dense complexes do not turn into a grey blob
NETWORK_NODE_PX = 14           # node diameter in Cytoscape pixels (the Python figure matches it)

# Sequential blue ramp (light -> dark), e.g. betweenness on the overview network
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ---------------------------------------------------------------------------
# 4. Ink, surface, typography, sizes
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE = 8                  # body text in figures (pt)
TITLE_SIZE = 9
PANEL_LABEL_SIZE = 11          # the A, B, C letters

FULL_WIDTH = 6.3               # inches: text width of an A4 page with 2.5 cm margins
HALF_WIDTH = 3.1
DPI = 300
FORMATS = ["png", "pdf", "svg"]  # pdf/svg keep text editable (Inkscape, Illustrator, PowerPoint)


def _font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT_CANDIDATES if f in available), "DejaVu Sans")


def apply():
    """Set matplotlib defaults so every figure shares the same look."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_font()],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": FONT_SIZE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",      # text stays text in SVG
        "pdf.fonttype": 42,          # editable TrueType text in PDF
    })


def panel_label(ax, letter, x=-0.08, y=1.04):
    """Bold A/B/C label in the top-left corner of a panel."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight="bold", color=INK, va="bottom", ha="right")


def save(fig, path_without_ext):
    """Save one figure in every format listed in FORMATS."""
    for ext in FORMATS:
        fig.savefig(f"{path_without_ext}.{ext}", dpi=DPI)


def place_labels(ax, items, fontsize=None, weight="bold", distance=6):
    """Add text labels next to points without letting them overlap each other.

    items: list of (x, y, text), most important first. For each label, eight
    positions around the point are tried and the first free one is used.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fontsize = fontsize or FONT_SIZE - 1
    d = distance
    offsets = [(0, d, "center", "bottom"), (d, d, "left", "bottom"), (d, 0, "left", "center"),
               (d, -d, "left", "top"), (0, -d, "center", "top"), (-d, -d, "right", "top"),
               (-d, 0, "right", "center"), (-d, d, "right", "bottom")]
    placed = []
    for x, y, text in items:
        best = None
        for dx, dy, ha, va in offsets:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                              fontsize=fontsize, fontweight=weight, color=INK, zorder=6,
                              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            bb = ann.get_window_extent(renderer).expanded(1.05, 1.1)
            overlap = sum(bb.overlaps(p) for p in placed)
            if best is None or overlap < best[0]:
                if best is not None:
                    best[1].remove()
                best = (overlap, ann, bb)
            else:
                ann.remove()
            if overlap == 0:
                break
        placed.append(best[2])
