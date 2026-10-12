# =============================================================================
# REPOSITORY BANNER: THE PROBLEM AND THE OPERATOR, DRAWN FROM THE REAL DATA
#
#   python run.py generate --cases 500        (once; the 500-case dataset)
#   python docs/figures/make_banner.py        (from the project folder)
#
#   (a) geometry and boundary conditions (line drawing)
#   (b) the finite element mesh of that case (Netgen, as stored)
#   (c) the network input: signed distance field on the 128 x 96 grid
#   (d) the network output: temperature on the same grid (NGSolve solution)
#   bottom strip: six other cases of the dataset, same colour scale
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.tri as mtri

for path in font_manager.findSystemFonts():
    if "lmroman10-regular" in path or "lmroman10-italic" in path:
        font_manager.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm",
                     "font.size": 11, "axes.linewidth": 0.6})

ROOT = "data/processed/combined_0500/protocol_v2/dataset.npz"
FE = "data/raw/combined_0500/protocol_v2/fe_cases/fe_case_%06d.npz"
HERO = 0
STRIP = [6, 12, 10, 32, 21, 23]

data = np.load(ROOT)
Lx, Ly = 36.0, 21.0
NX, NY = 128, 96
T_all = data["physical_temp"]
meta = data["meta"]

# ONE WARM SEQUENTIAL SCALE FOR EVERY TEMPERATURE PANEL (light = cool)
heat = LinearSegmentedColormap.from_list(
    "heat", ["#fbf6ec", "#f3d9a8", "#e4a35e", "#c4602f", "#8a2a1b", "#3d0f0b"])
T_LOW, T_HIGH = 300.0, 1000.0


def grid_field(values):
    return values.reshape(NY, NX)


def cavity_patch(index, **style):
    shape, p1, p2, cx, cy = meta[index]
    cx, cy, p1, p2 = cx * 1e3, cy * 1e3, p1 * 1e3, p2 * 1e3
    if int(shape) == 0:
        return Circle((cx, cy), p1 / 2, **style)
    return Rectangle((cx - p1 / 2, cy - p2 / 2), p1, p2, **style)


def plate_axes(axis):
    axis.set_xlim(-0.6, Lx + 0.6)
    axis.set_ylim(-0.6, Ly + 0.6)
    axis.set_aspect("equal")
    axis.axis("off")


def temperature_panel(axis, index, isotherms=True):
    T = grid_field(T_all[index]).astype(float)
    mask = grid_field(data["mask"][index]) > 0.5
    T = np.where(mask, T, np.nan)
    x = np.linspace(0, Lx, NX)
    y = np.linspace(0, Ly, NY)
    axis.pcolormesh(x, y, T, cmap=heat, vmin=T_LOW, vmax=T_HIGH, shading="gouraud",
                    rasterized=True)
    if isotherms:
        axis.contour(x, y, T, levels=np.arange(350, 1100, 50), colors="k",
                     linewidths=0.35, alpha=0.55)
    axis.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, lw=0.8, ec="k"))
    axis.add_patch(cavity_patch(index, fill=False, lw=0.8, ec="k"))
    plate_axes(axis)


fig = plt.figure(figsize=(13.6, 4.3), dpi=200)
fig.patch.set_facecolor("white")

# PANEL POSITIONS [left, bottom, width, height] in figure fractions
top_y, top_h = 0.39, 0.58
w = 0.19
xa, xb, xc, xd = 0.025, 0.255, 0.485, 0.785

# -----------------------------------------------------------------------------
# (a) GEOMETRY AND BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
ax = fig.add_axes([xa, top_y, w, top_h])
ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, lw=1.0, ec="k"))
ax.add_patch(cavity_patch(HERO, fill=True, fc="white", ec="k", lw=1.0, hatch="////"))
# adiabatic edges: short ticks outside the left, right and bottom edges
for y in np.linspace(0.8, Ly - 0.8, 12):
    ax.plot([-0.0, -0.9], [y, y - 0.9], color="k", lw=0.5)
    ax.plot([Lx, Lx + 0.9], [y, y - 0.9], color="k", lw=0.5)
for x in np.linspace(1.0, Lx - 1.0, 18):
    ax.plot([x, x - 0.9], [0, -0.9], color="k", lw=0.5)
# heat flux: arrows onto the top edge
for x in np.linspace(3, Lx - 3, 9):
    ax.annotate("", xy=(x, Ly), xytext=(x, Ly + 4.2),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k", mutation_scale=7))
ax.text(Lx / 2, Ly + 5.2, r"$q$", ha="center", va="bottom", fontsize=13)
shape, p1, p2, cx, cy = meta[HERO]
ax.text(cx * 1e3, cy * 1e3 - p1 * 1e3 / 2 - 1.0, r"$h,\ T_\infty$", ha="center", va="top",
        fontsize=11)
ax.text(27.0, 15.5, r"$-\nabla\!\cdot(k\nabla T)=0$", ha="center", va="center", fontsize=10.5)
ax.text(-2.2, Ly / 2, r"$\partial_n T = 0$", ha="right", va="center", fontsize=10,
        rotation=90)
ax.set_xlim(-4.0, Lx + 1.5)
ax.set_ylim(-2.0, Ly + 7.5)
ax.set_aspect("equal")
ax.axis("off")

# -----------------------------------------------------------------------------
# (b) FINITE ELEMENT MESH OF THE SAME CASE
# -----------------------------------------------------------------------------
fe = np.load(str(data["fe_paths"][HERO]))
pts = fe["points"] * 1e3
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], fe["triangles"])
ax = fig.add_axes([xb, top_y, w, top_h])
ax.triplot(tri, color="k", lw=0.22)
plate_axes(ax)
ax.set_ylim(-2.0, Ly + 7.5)

# -----------------------------------------------------------------------------
# (c) INPUT: SIGNED DISTANCE FIELD ON THE GRID (level sets; phi = 0 bold)
# -----------------------------------------------------------------------------
sdf = grid_field(data["sdf"][HERO]) * Lx          # stored value / 36 mm  ->  mm
x = np.linspace(0, Lx, NX)
y = np.linspace(0, Ly, NY)
ax = fig.add_axes([xc, top_y, w, top_h])
gray = LinearSegmentedColormap.from_list("g", ["#e9e9e9", "#ffffff"])
ax.pcolormesh(x, y, sdf, cmap=gray, vmin=-6, vmax=18, shading="gouraud", rasterized=True)
ax.contour(x, y, sdf, levels=np.arange(1.5, 20, 1.5), colors="k", linewidths=0.35)
ax.contour(x, y, sdf, levels=[-3.0, -1.5], colors="k", linewidths=0.35, linestyles="--")
ax.contour(x, y, sdf, levels=[0.0], colors="k", linewidths=1.0)
ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, lw=0.8, ec="k"))
plate_axes(ax)
ax.set_ylim(-2.0, Ly + 7.5)

# -----------------------------------------------------------------------------
# (d) OUTPUT: TEMPERATURE ON THE GRID
# -----------------------------------------------------------------------------
ax = fig.add_axes([xd, top_y, w, top_h])
temperature_panel(ax, HERO)
ax.set_ylim(-2.0, Ly + 7.5)

# PANEL CAPTIONS (below each panel, never on the drawing)
cap_y = top_y - 0.015
captions = [
    (xa, "(a)  plate 36 × 21 mm, one cooled cavity"),
    (xb, "(b)  finite element mesh (NGSolve, order 2)"),
    (xc, r"(c)  input: signed distance $\phi(\mathbf{x})$, 128 × 96"),
    (xd, r"(d)  output: temperature $T(\mathbf{x})$, 128 × 96"),
]
for x0, text in captions:
    fig.text(x0 + w / 2, cap_y, text, ha="center", va="top", fontsize=10)

# ARROWS BETWEEN PANELS
def link(x0, x1, label=None):
    yy = top_y + top_h * 0.40
    fig.patches.append(FancyArrowPatch((x0, yy), (x1, yy), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=10, lw=0.8,
                                       color="k"))
    if label:
        fig.text((x0 + x1) / 2, yy + 0.035, label, ha="center", va="bottom", fontsize=10)

link(xa + w + 0.006, xb - 0.006)
link(xb + w + 0.006, xc - 0.006)

# THE OPERATOR, BETWEEN (c) AND (d)
gx0, gx1 = xc + w + 0.008, xd - 0.008
yy = top_y + top_h * 0.40
box_w, box_h = 0.075, 0.17
bx = (gx0 + gx1) / 2 - box_w / 2
fig.patches.append(Rectangle((bx, yy - box_h / 2), box_w, box_h, transform=fig.transFigure,
                             fill=False, lw=0.8, ec="k"))
fig.text(bx + box_w / 2, yy + 0.03, r"$\mathcal{G}_\theta$", ha="center", va="center",
         fontsize=17)
fig.text(bx + box_w / 2, yy - 0.045, "DeepONet\nor FNO", ha="center", va="center",
         fontsize=9, linespacing=1.1)
fig.patches.append(FancyArrowPatch((gx0, yy), (bx, yy), transform=fig.transFigure,
                                   arrowstyle="-", lw=0.8, color="k"))
fig.patches.append(FancyArrowPatch((bx + box_w, yy), (gx1, yy), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=10, lw=0.8, color="k"))

# -----------------------------------------------------------------------------
# BOTTOM STRIP: OTHER CASES OF THE DATASET (same colour scale)
# -----------------------------------------------------------------------------
n = len(STRIP)
sw = 0.118
gap = (0.80 - n * sw) / (n - 1)
for k, index in enumerate(STRIP):
    ax = fig.add_axes([0.025 + k * (sw + gap), 0.03, sw, 0.235])
    temperature_panel(ax, index, isotherms=True)

# COLOUR SCALE, to the right of the strip
cax = fig.add_axes([0.875, 0.075, 0.085, 0.022])
sm = plt.cm.ScalarMappable(cmap=heat, norm=plt.Normalize(T_LOW, T_HIGH))
bar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[300, 650, 1000])
bar.outline.set_linewidth(0.5)
bar.ax.tick_params(labelsize=8.5, width=0.5, length=2.5)
fig.text(0.9175, 0.115, r"$T$ (K)", ha="center", va="bottom", fontsize=9.5)
fig.text(0.9175, 0.205, "other cases\nof the dataset", ha="center", va="center",
         fontsize=9, linespacing=1.15)

fig.savefig("docs/figures/banner.png", dpi=200, facecolor="white")
fig.savefig("docs/figures/banner.svg", facecolor="white")
print("saved")
