# =============================================================================
# TEST: METRICS (analysis/metrics.py) AND TUNING GRIDS (training/tuning.py)
# =============================================================================

import numpy as np

from src.config import Settings
from src.analysis.metrics import compute_metrics
from src.training.tuning import grid_cells, cell_settings


def test_metrics_on_known_numbers():
    true = np.array([400.0, 500.0, 600.0])
    pred = np.array([401.0, 499.0, 603.0])
    m = compute_metrics(true, pred, T_inf=300.0)
    assert abs(m["rmse_K"] - np.sqrt(11.0 / 3.0)) < 1e-12
    assert abs(m["mae_K"] - 5.0 / 3.0) < 1e-12
    assert m["max_abs_K"] == 3.0
    assert abs(m["relative_l2"] - np.sqrt(11.0) / np.sqrt(400**2 + 500**2 + 600**2)) < 1e-15
    assert abs(m["relative_l2_rise"] - np.sqrt(11.0) / np.sqrt(100**2 + 200**2 + 300**2)) < 1e-15
    assert abs(m["r2"] - (1.0 - 11.0 / 20000.0)) < 1e-12


def test_deeponet_grid_is_5_by_5_on_branch_and_trunk():
    cfg = Settings()
    cells = grid_cells(cfg, "deeponet", "grid")
    assert len(cells) == 25
    names = set()
    for cell in cells:
        L, N = cell["x"], cell["y"]
        assert cell["fields"]["branch_hidden"] == (N,) * L
        assert cell["fields"]["trunk_hidden"] == (N,) * L
        assert cell["fields"]["latent_dim"] == 128
        names.add(cell_settings(cfg, "deeponet", "grid", cell, 123).run_name)
    assert len(names) == 25


def test_fno_first_stage():
    cells = grid_cells(Settings(), "fno", "modes_width")
    assert len(cells) == 12
    for cell in cells:
        assert cell["fields"]["fno_blocks"] == 4
