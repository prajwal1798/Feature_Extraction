# =============================================================================
# TEST: SIGNED DISTANCE FUNCTIONS AND CAVITY MESH  (fem/geometry.py)
# The rectangle points are the hand-worked, non-symmetric examples:
#   cx = 17.3 mm, cy = 8.6 mm, W = 9.4 mm, H = 5.2 mm
# =============================================================================

import types

import numpy as np

from src.fem.geometry import sdf_circle, sdf_rectangle, sdf_rounded_rectangle, build_mesh

MM = 1.0e-3


def test_rectangle_sdf_hand_points():
    cx, cy, W, H = 17.3 * MM, 8.6 * MM, 9.4 * MM, 5.2 * MM
    points = [((19.1, 9.9), -1.3),    # inside, nearest wall = top
              ((10.4, 7.5), 2.2),     # beside the left wall
              ((11.0, 3.1), 3.3121),  # diagonal to the bottom-left corner
              ((20.5, 4.2), 1.8)]     # below the bottom wall
    for (x, y), expected in points:
        phi = sdf_rectangle(x * MM, y * MM, cx, cy, W, H)
        assert abs(phi / MM - expected) < 1.0e-4, (x, y, phi / MM, expected)


def test_rectangle_sdf_zero_on_wall_and_symmetric():
    cx, cy, W, H = 20 * MM, 10 * MM, 12 * MM, 8 * MM
    assert abs(sdf_rectangle(26 * MM, 10 * MM, cx, cy, W, H)) < 1e-15
    a = sdf_rectangle(29 * MM, 17 * MM, cx, cy, W, H)
    b = sdf_rectangle(11 * MM, 3 * MM, cx, cy, W, H)
    assert abs(a - b) < 1e-15 and abs(a / MM - np.sqrt(18.0)) < 1e-9


def test_circle_sdf():
    assert abs(sdf_circle(0.0, 0.0, 0.0, 0.0, 2.0) + 1.0) < 1e-15   # centre: -radius
    assert abs(sdf_circle(3.0, 4.0, 0.0, 0.0, 2.0) - 4.0) < 1e-15   # 5 - 1


def test_rounded_rectangle_equals_rectangle_when_radius_is_zero():
    x = np.linspace(0, 0.036, 40)
    y = np.linspace(0, 0.021, 30)
    X, Y = np.meshgrid(x, y)
    a = sdf_rectangle(X, Y, 0.02, 0.01, 0.012, 0.008)
    b = sdf_rounded_rectangle(X, Y, 0.02, 0.01, 0.012, 0.008, 0.0)
    assert np.allclose(a, b, atol=1e-14)


def test_mesh_has_named_cavity_boundary():
    cfg = types.SimpleNamespace(Lx=0.036, Ly=0.021, maxh=0.0012, rounded_corner_segments=12)
    for shape, p1, p2, p3 in [("circle", 0.008, 0, 0), ("rect", 0.012, 0.008, 0),
                              ("rounded_rect", 0.012, 0.008, 0.002)]:
        mesh = build_mesh(cfg, shape, p1, p2, p3, 0.018, 0.0105)
        assert mesh.ne > 0
        assert "hole" in mesh.GetBoundaries()
        for name in ("bottom", "right", "top", "left"):
            assert name in mesh.GetBoundaries()
