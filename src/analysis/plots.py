# =============================================================================
# PLOTS: DATA FIGURES AND FIGURES OF ONE TRAINED RUN
#
#   plot_preview        one FE case: geometry, mesh, FE field, SDF, mask,
#                       grid temperature                    (data generation)
#   plot_sampling_space where the training cavities lie and how big they are
#   plot_loss           training / validation loss of one run
#   plot_case           reference, prediction and signed error of one case
#
# Study-level figures (several runs, seeds, operators, tuning grids) are in
# analysis/report.py.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   shape, p1, p2, p3, cx, cy : cavity description (see fem/geometry.py)
#   Xg, Yg      : grid coordinates, shape (Ny, Nx)                       [m]
#   sdf, mask, T: SDF [m], solid mask, grid temperature [K], shape (Ny, Nx)
#   pts, triangles, T_fe : FE vertices, triangles, vertex temperatures
#   meta        : per case [shape_code, p1, p2, cx, cy]
#   true, pred  : reference and predicted temperature of one case, (Ny*Nx,)
#   valid       : solid points of that case (boolean, (Ny*Nx,))
# =============================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .style import paper_style, save_figure, MODEL_NAMES


# =============================================================================
# CAVITY OUTLINE AS A CLOSED LINE (FOR DRAWING ONLY)
# =============================================================================
def cavity_outline(shape, p1, p2, cx, cy, p3=0.0):
    if shape == "circle":
        radius = 0.5 * p1
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        return x, y

    if shape == "rounded_rect":
        radius = min(p3, 0.5 * min(p1, p2) - 1.0e-12)
        a = 0.5 * p1 - radius
        b = 0.5 * p2 - radius
        corners = [
            (cx + a, cy - b, -0.5 * np.pi, 0.0),
            (cx + a, cy + b, 0.0, 0.5 * np.pi),
            (cx - a, cy + b, 0.5 * np.pi, np.pi),
            (cx - a, cy - b, np.pi, 1.5 * np.pi),
        ]
        x = []
        y = []
        for x0, y0, theta_start, theta_end in corners:
            for theta in np.linspace(theta_start, theta_end, 49):
                x.append(x0 + radius * np.cos(theta))
                y.append(y0 + radius * np.sin(theta))
        x.append(x[0])
        y.append(y[0])
        return np.array(x), np.array(y)

    # SHARP RECTANGLE
    left = cx - 0.5 * p1
    right = cx + 0.5 * p1
    bottom = cy - 0.5 * p2
    top = cy + 0.5 * p2
    x = np.array([left, right, right, left, left])
    y = np.array([bottom, bottom, top, top, bottom])
    return x, y


def plate_axis(ax, cfg):
    ax.plot([0.0, cfg.Lx, cfg.Lx, 0.0, 0.0], [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0], "k-", lw=1.5)
    ax.set_xlim(0.0, cfg.Lx)
    ax.set_ylim(0.0, cfg.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# =============================================================================
# PREVIEW OF ONE FE CASE (2 x 3 PANELS)
# =============================================================================
def plot_preview(cfg, shape, p1, p2, cx, cy, sdf, mask, T, pts, triangles, T_fe, p3=0.0):
    x_line = np.linspace(0.0, cfg.Lx, cfg.Nx)
    y_line = np.linspace(0.0, cfg.Ly, cfg.Ny)
    Xg, Yg = np.meshgrid(x_line, y_line, indexing="xy")
    outline_x, outline_y = cavity_outline(shape, p1, p2, cx, cy, p3)

    with paper_style():
        fig, ax = plt.subplots(2, 3, figsize=(18, 8), layout="constrained")

        # GEOMETRY
        plate_axis(ax[0, 0], cfg)
        ax[0, 0].plot(outline_x, outline_y, "r-", lw=2)
        ax[0, 0].set_title("Geometry")

        # MESH
        triangulation = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)
        ax[0, 1].triplot(triangulation, lw=0.4, color="tab:blue")
        plate_axis(ax[0, 1], cfg)
        ax[0, 1].set_title("FE mesh (%d triangles)" % len(triangles))

        # FE TEMPERATURE AT THE VERTICES
        image = ax[0, 2].tripcolor(triangulation, T_fe, shading="gouraud", cmap="inferno")
        ax[0, 2].plot(outline_x, outline_y, "w-", lw=1.5)
        plate_axis(ax[0, 2], cfg)
        ax[0, 2].set_title("FE temperature: %.1f .. %.1f K" % (T_fe.min(), T_fe.max()))
        fig.colorbar(image, ax=ax[0, 2], label="T [K]")

        # SIGNED DISTANCE
        image = ax[1, 0].contourf(Xg, Yg, sdf, 50, cmap="RdBu_r")
        ax[1, 0].contour(Xg, Yg, sdf, levels=[0.0], colors="k", linewidths=1.5)
        plate_axis(ax[1, 0], cfg)
        ax[1, 0].set_title("Signed distance")
        fig.colorbar(image, ax=ax[1, 0], label="SDF [m]")

        # MASK
        image = ax[1, 1].imshow(mask, origin="lower", extent=[0, cfg.Lx, 0, cfg.Ly],
                                interpolation="nearest", cmap="gray")
        ax[1, 1].contour(Xg, Yg, sdf, levels=[0.0], colors="r", linewidths=1.5)
        plate_axis(ax[1, 1], cfg)
        ax[1, 1].set_title("Solid mask (1 = solid)")

        # GRID TEMPERATURE (TRAINING TARGET)
        image = ax[1, 2].contourf(Xg, Yg, T, 50, cmap="inferno")
        ax[1, 2].contour(Xg, Yg, sdf, levels=[0.0], colors="w", linewidths=1.5)
        plate_axis(ax[1, 2], cfg)
        ax[1, 2].set_title("Grid temperature: %.1f .. %.1f K" % (np.nanmin(T), np.nanmax(T)))
        fig.colorbar(image, ax=ax[1, 2], label="T [K]")

        if cfg.show_figures:
            plt.show()
        save_figure(fig, Path(cfg.outdir) / "preview_figures" / "preview_case")


# =============================================================================
# SAMPLING SPACE OF THE TRAINING DATASET
#
# (a) circle centres coloured by D   (b) rectangle centres coloured by W*H
# (c) histogram of D                 (d) W against H
# =============================================================================
def plot_sampling_space(cfg, data):
    meta = data["meta"]
    circles = meta[:, 0] == 0
    rects = meta[:, 0] == 1

    with paper_style():
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

        if circles.any():
            points = ax[0, 0].scatter(meta[circles, 3], meta[circles, 4], c=meta[circles, 1] * 1e3, s=14)
            fig.colorbar(points, ax=ax[0, 0], label="D [mm]")
            ax[1, 0].hist(meta[circles, 1] * 1e3, bins=25, color="0.4")
        plate_axis(ax[0, 0], cfg)
        ax[0, 0].set_title("(a) Circle centres (%d)" % circles.sum())
        ax[1, 0].set_xlabel("D [mm]")
        ax[1, 0].set_ylabel("Count")
        ax[1, 0].set_title("(c) Circle diameters")

        if rects.any():
            area = meta[rects, 1] * meta[rects, 2] * 1e6
            points = ax[0, 1].scatter(meta[rects, 3], meta[rects, 4], c=area, s=14)
            fig.colorbar(points, ax=ax[0, 1], label="W H [mm$^2$]")
            ax[1, 1].scatter(meta[rects, 1] * 1e3, meta[rects, 2] * 1e3, s=12, color="0.3")
        plate_axis(ax[0, 1], cfg)
        ax[0, 1].set_title("(b) Rectangle centres (%d)" % rects.sum())
        ax[1, 1].set_xlabel("W [mm]")
        ax[1, 1].set_ylabel("H [mm]")
        ax[1, 1].set_title("(d) Rectangle sizes")

        folder = cfg._root(cfg.figures_root) / cfg.case_tag / "dataset"
        save_figure(fig, folder / "sampling_space")


# =============================================================================
# LOSS HISTORY OF ONE RUN (masked MSE of the scaled temperature)
# =============================================================================
def plot_loss(cfg):
    path = Path(cfg.log_outdir) / "loss_history.csv"
    history = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    epochs = history[:, 0]

    with paper_style():
        fig, ax = plt.subplots(figsize=(5.2, 3.7), layout="constrained")
        ax.semilogy(epochs, history[:, 1], color="blue", label="Train")
        ax.semilogy(epochs, history[:, 2], color="red", linestyle="--", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE (scaled T)")
        ax.legend()
        save_figure(fig, Path(cfg.figure_outdir) / "loss_history")


# =============================================================================
# ONE CASE: REFERENCE | PREDICTION | SIGNED ERROR (cavity left blank)
# =============================================================================
def plot_case(cfg, data, case_id, true, pred, valid, folder_name="test"):
    ny = int(data["Ny"][0])
    nx = int(data["Nx"][0])
    extent = [0, float(data["Lx"][0]), 0, float(data["Ly"][0])]

    reference = true.copy().reshape(ny, nx)
    prediction = pred.copy().reshape(ny, nx)
    error = (pred - true).reshape(ny, nx)

    cavity = ~valid.reshape(ny, nx)
    reference[cavity] = np.nan
    prediction[cavity] = np.nan
    error[cavity] = np.nan

    T_low = min(np.nanmin(reference), np.nanmin(prediction))
    T_high = max(np.nanmax(reference), np.nanmax(prediction))
    error_limit = max(float(np.nanmax(np.abs(error))), 1.0e-12)

    with paper_style():
        fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7), layout="constrained")

        panels = [
            (reference, "Reference (FE)", "viridis", T_low, T_high, "Temperature (K)"),
            (prediction, MODEL_NAMES.get(cfg.operator, "Prediction"), "viridis", T_low, T_high,
             "Temperature (K)"),
            (error, "Signed error", "RdBu_r", -error_limit, error_limit, "Error (K)"),
        ]
        for axis, (field, title, colormap, low, high, label) in zip(ax, panels):
            image = axis.imshow(field, origin="lower", interpolation="nearest", extent=extent,
                                cmap=colormap, vmin=low, vmax=high)
            axis.set_title(title)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            fig.colorbar(image, ax=axis, label=label)

        if cfg.show_figures:
            plt.show()
        save_figure(fig, Path(cfg.figure_outdir) / folder_name / ("case_%06d" % int(case_id)))
