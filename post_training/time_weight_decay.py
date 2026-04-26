"""
time_weight_decay.py
====================
Convenience re-exports and standalone decay-schedule utilities.

This file existed as an empty stub in the pipeline-build branch.
All core temporal logic lives in temporal_drag.py; import from there.

Quick re-exports
----------------
from post_training.time_weight_decay import (
    Time2Vec,
    TemporalEdgeInjector,
    TemporalDRAG,
    build_delta_t_edge_data,
    TemporalDataHandlerModule,
    TemporalModelHandlerModule,
    make_decay_schedule,
)
"""

from post_training.temporal_drag import (
    Time2Vec,
    TemporalEdgeInjector,
    TemporalDRAG,
    build_delta_t_edge_data,
    TemporalDataHandlerModule,
    TemporalModelHandlerModule,
    _temporal_weighted_ce,
)

import numpy as np
import torch


def make_decay_schedule(
    transaction_dt: np.ndarray,
    idx_train: list,
    decay: float = 0.5,
    scheme: str = "exponential",
) -> torch.Tensor:
    """
    Standalone utility to compute per-node temporal weights.

    Parameters
    ----------
    transaction_dt : (N,) array of raw TransactionDT seconds.
    idx_train      : list of training node indices.
    decay          : controls how fast older samples are down-weighted.
                     - exponential: half-life fraction of the time range.
                     - linear:      slope of the ramp (0 = flat, 1 = full ramp).
    scheme         : 'exponential' (default) or 'linear'.

    Returns
    -------
    weights : (N,) float tensor, mean weight of train nodes = 1.0.

    Examples
    --------
    >>> w = make_decay_schedule(dt_array, train_idx, decay=0.5, scheme="exponential")
    >>> w = make_decay_schedule(dt_array, train_idx, decay=1.0, scheme="linear")
    """
    N       = len(transaction_dt)
    dt      = torch.tensor(transaction_dt, dtype=torch.float32)
    weights = torch.ones(N, dtype=torch.float32)

    train_dt = dt[idx_train]
    t_min    = train_dt.min()
    t_range  = (train_dt.max() - t_min).clamp(min=1.0)

    # Normalised time in [0, 1] for training nodes
    t_norm   = (train_dt - t_min) / t_range

    if scheme == "exponential":
        w_train = torch.exp(t_norm / max(decay, 1e-6))
    elif scheme == "linear":
        w_train = 1.0 + decay * t_norm
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'exponential' or 'linear'.")

    # Normalise so mean = 1 (keeps loss scale stable)
    w_train = w_train / w_train.mean()
    weights[idx_train] = w_train

    return weights
