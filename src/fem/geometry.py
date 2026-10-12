# =============================================================================
# GEOMETRY: SIGNED DISTANCE FUNCTIONS AND FINITE ELEMENT MESHES
#
# One case = rectangular plate [0, Lx] x [0, Ly] with ONE internal cavity.
# This module answers two questions for a given cavity:
#   (1) What is the signed distance phi(x, y) on the learning grid?
#       -> input field of every neural operator
#   (2) What is the triangular mesh of the solid?
#       -> domain of the finite element solve
#
# Sign convention used everywhere in the project:
#   phi < 0  inside the cavity  (coolant, not solved)
#   phi = 0  on the cavity wall (Robin boundary "hole")
#   phi > 0  inside the solid   (conducting region)
# =============================================================================

# =============================================================================
# VARIABLES (meaning and units, listed once)
#
#   x, y      : query coordinates                      [m]  arrays of any shape
#   cx, cy    : cavity centre (changes from case to case)  [m]
#   D         : circle diameter                        [m]
#   W, H      : rectangle width and height             [m]
#   R         : corner radius of a rounded rectangle   [m]
#   Lx, Ly    : plate width and height  (cfg.Lx, cfg.Ly)  [m]
#   maxh      : maximum element size    (cfg.maxh)        [m]
#   Xg, Yg    : learning-grid coordinates, shape (Ny, Nx)
#   p1,p2,p3  : shape parameters in the dataset layout
#               circle       : p1 = D
#               rect         : p1 = W, p2 = H
#               rounded_rect : p1 = W, p2 = H, p3 = R
#   n_corner  : straight pieces used per rounded corner in the mesh
#
# Boundary names written into the mesh (used later by the solver):
#   "bottom", "right", "top", "left"  : outer plate edges
#   "hole"                            : complete cavity wall
# =============================================================================

import numpy as np
from ngsolve import Mesh
from netgen.geom2d import SplineGeometry


# =============================================================================
# SIGNED DISTANCE: CIRCLE
#
#   phi(x, y) = (distance from the centre) - (radius)
# =============================================================================
def sdf_circle(x, y, cx, cy, D):
    radius = 0.5 * D

    distance_to_centre = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    phi = distance_to_centre - radius

    return phi


# =============================================================================
# SIGNED DISTANCE: AXIS-ALIGNED RECTANGLE
#
# Step 1  measure from the cavity centre and drop the sign (folding):
#           u = |x - cx|,  v = |y - cy|
# Step 2  how far beyond the walls the point is:
#           dx = u - W/2   (> 0 : beyond the left/right walls)
#           dy = v - H/2   (> 0 : beyond the top/bottom walls)
# Step 3  outside the cavity : distance to nearest edge or corner
#           sqrt( max(dx,0)^2 + max(dy,0)^2 )
#         inside the cavity  : minus the distance to the nearest wall
#           min( max(dx,dy), 0 )
#         Only one of the two is non-zero, so phi = outside + inside.
# =============================================================================
def sdf_rectangle(x, y, cx, cy, W, H):
    half_W = 0.5 * W
    half_H = 0.5 * H

    # STEP 1: FOLD INTO ONE QUARTER
    u = np.abs(x - cx)
    v = np.abs(y - cy)

    # STEP 2: DISTANCE BEYOND THE WALLS (NEGATIVE = STILL INSIDE THAT PAIR OF WALLS)
    dx = u - half_W
    dy = v - half_H

    # STEP 3a: OUTSIDE PART (KEEP ONLY THE POSITIVE PARTS)
    dx_out = np.maximum(dx, 0.0)
    dy_out = np.maximum(dy, 0.0)
    outside = np.sqrt(dx_out ** 2 + dy_out ** 2)

    # STEP 3b: INSIDE PART (NEAREST WALL, ZERO WHEN OUTSIDE)
    nearest = np.maximum(dx, dy)
    inside = np.minimum(nearest, 0.0)

    phi = outside + inside
    return phi


# =============================================================================
# SIGNED DISTANCE: ROUNDED RECTANGLE (external test geometry only)
#
# A rounded rectangle = a smaller "core" rectangle grown outwards by R:
#   core width  = W - 2R,   core height = H - 2R
#   phi_rounded = phi_core - R
#
# R may not exceed half the shorter side, otherwise the core would have
# negative size; it is clipped just below that limit.
# =============================================================================
def sdf_rounded_rectangle(x, y, cx, cy, W, H, R):
    largest_radius = 0.5 * min(W, H) - 1.0e-12
    radius = min(R, largest_radius)

    half_W_core = 0.5 * W - radius
    half_H_core = 0.5 * H - radius

    u = np.abs(x - cx)
    v = np.abs(y - cy)

    dx = u - half_W_core
    dy = v - half_H_core

    dx_out = np.maximum(dx, 0.0)
    dy_out = np.maximum(dy, 0.0)
    outside = np.sqrt(dx_out ** 2 + dy_out ** 2)

    nearest = np.maximum(dx, dy)
    inside = np.minimum(nearest, 0.0)

    phi = outside + inside - radius
    return phi


# =============================================================================
# OUTER PLATE GEOMETRY (shared by every cavity type)
#
# Netgen rule: every boundary line has a "leftdomain" and a "rightdomain",
# judged while walking along the line in its direction.
#   domain 1 = conducting solid (meshed)
#   domain 0 = outside / void   (not meshed)
#
# AddRectangle walks counter-clockwise, so the plate is on the LEFT of every
# outer edge  ->  leftdomain = 1, rightdomain = 0.
# =============================================================================
def start_plate_geometry(cfg):
    geo = SplineGeometry()

    geo.AddRectangle(
        (0.0, 0.0),
        (cfg.Lx, cfg.Ly),
        leftdomain=1,
        rightdomain=0,
        bcs=["bottom", "right", "top", "left"],
    )

    return geo


# =============================================================================
# CAVITY BOUNDARY AS A CLOSED POLYGON
#
# The points must be ordered COUNTER-CLOCKWISE around the cavity. Walking
# counter-clockwise, the cavity is on the LEFT and the solid on the RIGHT
#   ->  leftdomain = 0 (void), rightdomain = 1 (solid).
# The last point is joined back to the first, closing the loop.
# =============================================================================
def add_polygon_hole(geo, points):
    # REGISTER EVERY CORNER WITH NETGEN AND KEEP THE NUMBER (ID) IT RETURNS
    point_ids = []
    for px, py in points:
        new_id = geo.AppendPoint(px, py)
        point_ids.append(new_id)

    n_points = len(point_ids)

    # JOIN POINT i TO POINT i+1; THE LAST PASS WRAPS BACK TO POINT 0
    for i in range(n_points):
        start = point_ids[i]
        end = point_ids[(i + 1) % n_points]
        geo.Append(["line", start, end], leftdomain=0, rightdomain=1, bc="hole")


# =============================================================================
# RECTANGLE CORNER POINTS (counter-clockwise)
#
#   4 (left, top) <------ 3 (right, top)
#        |                     ^
#        v                     |
#   1 (left, bottom) ---> 2 (right, bottom)
# =============================================================================
def rectangle_points(cx, cy, W, H):
    left = cx - 0.5 * W
    right = cx + 0.5 * W
    bottom = cy - 0.5 * H
    top = cy + 0.5 * H

    points = []
    points.append((left, bottom))
    points.append((right, bottom))
    points.append((right, top))
    points.append((left, top))

    return points


# =============================================================================
# ROUNDED RECTANGLE BOUNDARY POINTS (counter-clockwise)
#
# Each corner is a quarter circle of radius R. Arc centres sit at
#   (cx +- a, cy +- b),   a = W/2 - R,   b = H/2 - R.
# Corners are visited counter-clockwise; angles in radians:
#   bottom-right : -90 deg ..   0 deg
#   top-right    :   0 deg ..  90 deg
#   top-left     :  90 deg .. 180 deg
#   bottom-left  : 180 deg .. 270 deg
# The straight edges appear automatically between consecutive arcs.
#
# Each quarter circle is replaced by n_corner short straight pieces.
# Worst gap between arc and pieces: R * (1 - cos(pi / (4 n_corner)))
#   R = 3 mm, n_corner = 12  ->  about 0.026 mm (grid spacing ~ 0.28 mm).
# =============================================================================
def rounded_rectangle_points(cx, cy, W, H, R, n_corner):
    largest_radius = 0.5 * min(W, H) - 1.0e-12
    radius = min(R, largest_radius)

    a = 0.5 * W - radius
    b = 0.5 * H - radius

    if n_corner < 2:
        n_corner = 2

    # ARC CENTRE (x0, y0) AND START / END ANGLE OF EACH CORNER
    corners = []
    corners.append((cx + a, cy - b, -0.5 * np.pi, 0.0))      # bottom-right
    corners.append((cx + a, cy + b, 0.0, 0.5 * np.pi))       # top-right
    corners.append((cx - a, cy + b, 0.5 * np.pi, np.pi))     # top-left
    corners.append((cx - a, cy - b, np.pi, 1.5 * np.pi))     # bottom-left

    tolerance = 1.0e-14
    points = []

    for x0, y0, theta_start, theta_end in corners:
        angles = np.linspace(theta_start, theta_end, n_corner + 1)

        for theta in angles:
            px = x0 + radius * np.cos(theta)
            py = y0 + radius * np.sin(theta)

            # SKIP A POINT THAT SITS ON TOP OF THE PREVIOUS ONE
            # (happens when R is at its limit and a straight edge has zero length)
            if len(points) > 0:
                last_x, last_y = points[-1]
                gap = np.hypot(px - last_x, py - last_y)
                if gap <= tolerance:
                    continue

            points.append((px, py))

    # DROP THE LAST POINT IF IT SITS ON TOP OF THE FIRST (THE LOOP CLOSES ITSELF)
    if len(points) > 1:
        first_x, first_y = points[0]
        last_x, last_y = points[-1]
        gap = np.hypot(first_x - last_x, first_y - last_y)
        if gap < tolerance:
            points.pop()

    return points


# =============================================================================
# MESH: PLATE WITH ONE CAVITY
#
# Builds the plate, cuts the cavity out, then lets Netgen fill the solid with
# triangles no larger than maxh. The mesh follows the circle and rectangle
# walls exactly; the rounded rectangle uses the polygon above.
# =============================================================================
def build_mesh(cfg, shape, p1, p2, p3, cx, cy):
    geo = start_plate_geometry(cfg)

    if shape == "circle":
        D = p1
        radius = 0.5 * D
        # AddCircle walks counter-clockwise: cavity on the left -> domain 0
        geo.AddCircle((cx, cy), radius, leftdomain=0, rightdomain=1, bc="hole")

    elif shape == "rect":
        W = p1
        H = p2
        points = rectangle_points(cx, cy, W, H)
        add_polygon_hole(geo, points)

    elif shape == "rounded_rect":
        W = p1
        H = p2
        R = p3
        points = rounded_rectangle_points(cx, cy, W, H, R, cfg.rounded_corner_segments)
        add_polygon_hole(geo, points)

    else:
        raise ValueError("Unknown shape. Use 'circle', 'rect' or 'rounded_rect'.")

    netgen_mesh = geo.GenerateMesh(maxh=cfg.maxh)
    mesh = Mesh(netgen_mesh)

    return mesh


# =============================================================================
# SIGNED DISTANCE ON THE LEARNING GRID
# =============================================================================
def grid_sdf(shape, p1, p2, p3, cx, cy, Xg, Yg):
    if shape == "circle":
        D = p1
        phi = sdf_circle(Xg, Yg, cx, cy, D)

    elif shape == "rect":
        W = p1
        H = p2
        phi = sdf_rectangle(Xg, Yg, cx, cy, W, H)

    elif shape == "rounded_rect":
        W = p1
        H = p2
        R = p3
        phi = sdf_rounded_rectangle(Xg, Yg, cx, cy, W, H, R)

    else:
        raise ValueError("Unknown shape. Use 'circle', 'rect' or 'rounded_rect'.")

    return phi


# =============================================================================
# PUBLIC ENTRY POINT (called by fem/sampling.sample_case)
#
# Returns the FE mesh and the SDF on the learning grid for one case.
# The SDF is stored in single precision (float32), as used by the networks.
# =============================================================================
def build_mesh_and_sdf(cfg, shape, p1, p2, cx, cy, Xg, Yg, p3=0.0):
    mesh = build_mesh(cfg, shape, p1, p2, p3, cx, cy)

    phi = grid_sdf(shape, p1, p2, p3, cx, cy, Xg, Yg)
    phi = phi.astype(np.float32)

    return mesh, phi
