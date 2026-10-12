"""Training-only POD on a common Cartesian grid with moving cavities.

Temperatures at cavity points are missing observations, not zero-temperature
measurements. Extend each snapshot there using the training pointwise mean;
its centred extension is zero. This is a declared extension-based POD, not a
claim that masked SVD solves a weighted low-rank completion problem.
"""

import numpy as np


def fit_pod(temperature, mask, train_idx, max_modes=128):
    ids = np.asarray(train_idx, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("POD needs at least two distinct training cases")
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    values = np.asarray(temperature[ids], dtype=np.float64)
    valid = np.asarray(mask[ids]) > 0.5
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("Temperature and mask must have shape (cases, points)")
    if not np.isfinite(values[valid]).all() or not valid.any():
        raise ValueError("Invalid training solid temperatures")
    count = valid.sum(axis=0)
    global_mean = np.where(valid, values, 0).sum() / valid.sum()
    mean = np.divide(
        np.where(valid, values, 0).sum(axis=0), count,
        out=np.full(values.shape[1], global_mean), where=count > 0,
    )
    centred = np.where(valid, values - mean, 0.0)
    # Method of snapshots: avoid a points-by-points covariance matrix.
    eigenvalues, vectors = np.linalg.eigh(centred @ centred.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    vectors = vectors[:, order]
    tolerance = np.finfo(np.float64).eps * max(centred.shape) * eigenvalues[0]
    numerical_rank = int((eigenvalues > tolerance).sum())
    if numerical_rank == 0:
        raise ValueError("Training snapshots have no nonzero POD modes")
    retained = min(max_modes, numerical_rank)
    modes = (centred.T @ vectors[:, :retained]) / np.sqrt(eigenvalues[:retained])
    # Fix sign for repeatable serialization, then use RMS-normalized modes.
    for column in range(retained):
        pivot = np.argmax(np.abs(modes[:, column]))
        if modes[pivot, column] < 0:
            modes[:, column] *= -1
    modes *= np.sqrt(values.shape[1])
    return {
        "mean": mean.astype(np.float32),
        "basis": modes.astype(np.float32),
        "eigenvalues": eigenvalues.astype(np.float64),
        "cumulative_energy": (np.cumsum(eigenvalues) / eigenvalues.sum()),
        "train_idx": ids.copy(),
        "observed_counts": count,
        "numerical_rank": numerical_rank,
        "retained_modes": retained,
        "extension": "training_pointwise_mean_at_missing_cavity_points",
        "mode_scaling": "unit_rms_on_full_cartesian_grid",
    }


def projection_errors(temperature, mask, ids, mean, basis, progress=None):
    """Best masked least-squares errors in this fixed affine POD space.

Targets are used only for this oracle diagnostic, never for fitting the basis
or branch. This bounds achievable error for the fixed basis on each case.
"""
    rows = []
    mean = np.asarray(mean, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    for number, case_id in enumerate(ids, 1):
        valid = np.asarray(mask[case_id]) > 0.5
        target = np.asarray(temperature[case_id], dtype=np.float64)[valid]
        if not len(target) or not np.isfinite(target).all():
            raise ValueError("Invalid solid targets for projection diagnostic")
        coefficients, _, rank, _ = np.linalg.lstsq(
            basis[valid], target - mean[valid], rcond=None
        )
        residual = mean[valid] + basis[valid] @ coefficients - target
        rows.append({"case_id": int(case_id), "sse": float(residual @ residual),
                     "points": int(valid.sum()), "rank": int(rank),
                     "rmse": float(np.sqrt(np.mean(residual ** 2)))})
        if progress is not None and (number % 25 == 0 or number == len(ids)):
            progress(number, len(ids))
    return rows
