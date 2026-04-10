"""
temporal_drag.py
================
Temporal Weighting integration for the pipeline-build branch of DRAG.

What this file does
-------------------
1.  Time2Vec          – learnable sinusoidal time encoding for edge delta-t values.
2.  TemporalEdgeInjector – wraps any DRAGConv layer and adds a per-relation
                           temporal signal to the GATv2 output inside DGL's
                           local_scope, without touching layers.py.
3.  TemporalDRAG      – drop-in subclass of DRAG that replaces the inner
                           DRAGConv forward pass with the temporal-aware version.
4.  build_delta_t_edge_data  – utility that writes delta-t values onto the DGL
                           heterograph edges so they travel with the graph
                           (used in ieee_cis_dataset.py).
5.  TemporalDataHandlerModule – thin subclass of DataHandlerModule that adds
                           chronological train/val/test splits (60/20/20 by
                           TransactionDT) alongside the standard stratified
                           split already used by the pipeline.
6.  TemporalModelHandlerModule – thin subclass of ModelHandlerModule that
                           (a) swaps in TemporalDRAG,
                           (b) uses temporally-weighted CrossEntropyLoss so
                               recent transactions are up-weighted during
                               training.

Usage
-----
In run.py / main.ipynb, replace:

    from data_handler  import DataHandlerModule
    from model_handler import ModelHandlerModule

    dh = DataHandlerModule(config)
    mh = ModelHandlerModule(config, dh)

with:

    from post_training.temporal_drag import (
        TemporalDataHandlerModule,
        TemporalModelHandlerModule,
    )

    dh = TemporalDataHandlerModule(config)
    mh = TemporalModelHandlerModule(config, dh)

Add these keys to your JSON config (all are optional – defaults shown):

    "apply_temporal"  : true,       // master switch
    "time_enc_dim"    : 16,         // Time2Vec output dimension
    "temporal_decay"  : 0.5,        // half-life fraction of the time range
                                    // used to build sample weights (0 = off)
    "temporal_split"  : false       // true  → chronological 60/20/20 split
                                    // false → keep existing stratified split
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# ─── Re-use existing pipeline classes ────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models        import DRAG
from layers        import DRAGConv
# from data_handler  import DataHandlerModule
# from model_handler import ModelHandlerModule


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Time2Vec
# ═════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    Learnable time encoding: one linear term + (out_dim-1) sinusoidal terms.

    Input  : t  – shape (E,)  raw delta-t values (seconds)
    Output : shape (E, out_dim)
    """

    def __init__(self, out_dim: int):
        super().__init__()
        assert out_dim >= 2, "Time2Vec needs at least 2 dimensions"
        self.out_dim = out_dim
        # Linear component
        self.w_linear = nn.Parameter(torch.empty(1))
        self.b_linear = nn.Parameter(torch.empty(1))
        # Periodic components
        self.w_period = nn.Parameter(torch.empty(out_dim - 1))
        self.b_period = nn.Parameter(torch.empty(out_dim - 1))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.w_linear, 0.0, 1.0)
        nn.init.uniform_(self.b_linear, 0.0, 1.0)
        nn.init.uniform_(self.w_period, 0.0, 2 * math.pi)
        nn.init.uniform_(self.b_period, 0.0, 2 * math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (E,)  →  unsqueeze to (E,1) for broadcasting
        t = t.unsqueeze(-1)                                     # (E, 1)
        linear   = self.w_linear * t + self.b_linear            # (E, 1)
        periodic = torch.sin(self.w_period * t + self.b_period) # (E, out_dim-1)
        return torch.cat([linear, periodic], dim=-1)            # (E, out_dim)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TemporalEdgeInjector
# ═════════════════════════════════════════════════════════════════════════════

class TemporalEdgeInjector(nn.Module):
    """
    Wraps one DRAGConv layer and adds a temporal bias.

    For each relation (edge type):
      1. Read the pre-computed 'delta_t' edge feature from the DGL block.
      2. Encode it with Time2Vec  →  (E, time_enc_dim).
      3. Project to (E, out_dim) with a learned linear layer.
      4. Scatter-mean the per-edge projections to destination nodes.
      5. Add the aggregated temporal signal to the GATv2 output.

    The original DRAGConv is unchanged; we call it first then add the bias.
    """

    def __init__(
        self,
        drag_conv: DRAGConv,
        out_dim: int,
        rel_names: list[str],
        time_enc_dim: int = 16,
    ):
        super().__init__()
        self.drag_conv  = drag_conv          # the original layer
        self.out_dim    = out_dim
        self.rel_names  = rel_names
        self.time2vec   = Time2Vec(time_enc_dim)

        # One projection per relation (same pattern as your notebook's W_time)
        self.W_time = nn.ModuleDict({
            rel: nn.Linear(time_enc_dim, out_dim, bias=False)
            for rel in rel_names
        })
        self._init_weights()

    def _init_weights(self):
        for layer in self.W_time.values():
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, graph, feat, get_attention=False):
        """
        graph : a DGL block (one relation at a time, as DRAG already does).
        feat  : node features (N, in_dim).
        Returns the same shape as DRAGConv.forward().
        """
        # ── Step A: original GATv2 pass ──────────────────────────────────────
        if get_attention:
            h, attn = self.drag_conv(graph, feat, get_attention=True)
        else:
            h = self.drag_conv(graph, feat, get_attention=False)

        # h shape: (N_dst, num_heads, out_dim//num_heads)  [concat mode]
        # We need to add a temporal bias of shape (N_dst, num_heads, out_dim//num_heads)
        # so we compute the bias in the flat (N_dst, out_dim) space, then reshape.

        # ── Step B: read delta_t from edge data ──────────────────────────────
        # The edge feature 'delta_t' is written by build_delta_t_edge_data().
        # If it's missing (e.g. non-IEEE-CIS datasets), skip silently.
        if "delta_t" not in graph.edata:
            if get_attention:
                return h, attn
            return h

        delta_t = graph.edata["delta_t"].float()   # (E,)

        # ── Step C: determine which relation this block belongs to ────────────
        # DGL blocks carry the etype; pick the first one (blocks are per-type).
        etypes = graph.canonical_etypes          # list of (src, rel, dst) triples
        if len(etypes) == 1:
            rel_key = etypes[0][1]               # middle element is the rel name
        else:
            # heterogeneous block – fall back to first etype
            rel_key = etypes[0][1]

        if rel_key not in self.W_time:
            if get_attention:
                return h, attn
            return h

        # ── Step D: encode & project ─────────────────────────────────────────
        t_enc  = self.time2vec(delta_t)           # (E, time_enc_dim)
        t_proj = self.W_time[rel_key](t_enc)      # (E, out_dim)

        # ── Step E: scatter-mean to destination nodes ─────────────────────────
        N_dst  = graph.number_of_dst_nodes()
        dst_idx = graph.edges()[1]                # (E,)  destination node indices

        t_agg  = torch.zeros(N_dst, self.out_dim, device=feat.device)
        count  = torch.zeros(N_dst, 1,            device=feat.device)

        t_agg.scatter_add_(
            0,
            dst_idx.unsqueeze(1).expand_as(t_proj),
            t_proj
        )
        count.scatter_add_(
            0,
            dst_idx.unsqueeze(1),
            torch.ones(len(dst_idx), 1, device=feat.device)
        )
        t_agg = t_agg / count.clamp(min=1.0)     # (N_dst, out_dim)

        # ── Step F: add temporal bias to GATv2 output ────────────────────────
        # h may be (N_dst, num_heads, head_dim) – reshape to flat, add, reshape back
        h_shape = h.shape
        h_flat  = h.reshape(N_dst, -1)            # (N_dst, out_dim)

        # Only add where t_agg rows are non-zero (nodes with edges)
        h_flat  = h_flat + t_agg
        h       = h_flat.reshape(h_shape)

        if get_attention:
            return h, attn
        return h


# ═════════════════════════════════════════════════════════════════════════════
# 3.  TemporalDRAG
# ═════════════════════════════════════════════════════════════════════════════

class TemporalDRAG(DRAG):
    """
    Drop-in replacement for DRAG.  Identical architecture + temporal injection.

    All constructor arguments are forwarded to DRAG unchanged, so you can
    replace DRAG(...) with TemporalDRAG(..., time_enc_dim=16) everywhere.
    """

    def __init__(self, *args, time_enc_dim: int = 16, **kwargs):
        super().__init__(*args, **kwargs)

        # Collect all relation names from all layers
        # DRAG stores layers as self.layers[layer_i][relation_j]
        # The relation names are the etypes of the graph – we need them at
        # construction time.  We derive them from num_relations and pass them
        # in via a new kwarg `rel_names`.
        rel_names = kwargs.get("rel_names", None)
        if rel_names is None:
            # Fallback: use generic names; temporal injection will match by
            # position.  Override by passing rel_names=[...] explicitly.
            rel_names = [f"rel_{j}" for j in range(self.num_relations)]

        self._rel_names    = rel_names
        self._time_enc_dim = time_enc_dim

        # Wrap every DRAGConv with a TemporalEdgeInjector
        for i in range(self.num_layers):
            out_dim = self.emb_dim[i]
            for j in range(self.num_relations):
                original_conv = self.layers[i][j]
                rel_name      = rel_names[j] if j < len(rel_names) else f"rel_{j}"
                self.layers[i][j] = TemporalEdgeInjector(
                    drag_conv    = original_conv,
                    out_dim      = out_dim,
                    rel_names    = [rel_name],
                    time_enc_dim = time_enc_dim,
                )

    # forward() is inherited unchanged from DRAG – it calls
    # self.layers[i][j](block, x) which now routes through TemporalEdgeInjector.


# ═════════════════════════════════════════════════════════════════════════════
# 4.  build_delta_t_edge_data
# ═════════════════════════════════════════════════════════════════════════════

# def build_delta_t_edge_data(graph: dgl.DGLGraph, transaction_dt: np.ndarray) -> dgl.DGLGraph:
#     """
#     Attach a 'delta_t' edge feature to every edge type in the DGL heterograph.

#     delta_t[e] = |TransactionDT[src(e)] - TransactionDT[dst(e)]|

#     Call this after building the graph in ieee_cis_dataset.py:

#         graph = build_delta_t_edge_data(graph, df["TransactionDT"].values)

#     Parameters
#     ----------
#     graph          : DGL heterograph (output of load_ieee_cis).
#     transaction_dt : (N,) array of raw TransactionDT seconds for each node.

#     Returns
#     -------
#     The same graph object with edata['delta_t'] set per edge type.
#     """
#     dt_tensor = torch.tensor(transaction_dt, dtype=torch.float32)

#     for etype in graph.canonical_etypes:
#         src_ids, dst_ids = graph.edges(etype=etype)
#         delta_t = (dt_tensor[src_ids] - dt_tensor[dst_ids]).abs()
#         graph.edges[etype].data["delta_t"] = delta_t

#     return graph


# ═════════════════════════════════════════════════════════════════════════════
# 5.  TemporalDataHandlerModule
# ═════════════════════════════════════════════════════════════════════════════

class TemporalDataHandlerModule(DataHandlerModule):
    """
    Extends DataHandlerModule with:
      - Optional chronological 60/20/20 split (set "temporal_split": true in config).
      - Exposes transaction_dt as self.dataset['transaction_dt'] for loss weighting.

    For IEEE-CIS the TransactionDT is stored as graph.ndata['transaction_dt']
    by the patched ieee_cis_dataset.py (see patch instructions at the bottom
    of this file).  For other datasets the chronological split is skipped.
    """

    def __init__(self, configuration):
        super().__init__(configuration)

        apply_temporal   = configuration.get("apply_temporal", True)
        temporal_split   = configuration.get("temporal_split", False)

        if not apply_temporal:
            return

        graph = self.dataset["graph"]

        # ── Extract TransactionDT if available ────────────────────────────────
        if "transaction_dt" in graph.ndata:
            dt = graph.ndata["transaction_dt"].numpy()   # (N,)
        else:
            dt = None

        self.dataset["transaction_dt"] = dt             # may be None

        # ── Chronological split (optional) ───────────────────────────────────
        if temporal_split and dt is not None:
            labels = self.dataset["labels"]
            n      = len(labels)
            order  = np.argsort(dt)

            n_train = int(0.60 * n)
            n_val   = int(0.80 * n)

            idx_train = order[:n_train].tolist()
            idx_valid = order[n_train:n_val].tolist()
            idx_test  = order[n_val:].tolist()

            y_train = labels[idx_train]
            y_valid = labels[idx_valid]
            y_test  = labels[idx_test]

            # Rebuild loaders with the new splits
            import dgl as _dgl
            emb_size = configuration["emb_size"]
            batch_size = configuration["batch_size"]
            sampler = _dgl.dataloading.MultiLayerFullNeighborSampler(len(emb_size))

            valid_loader = _dgl.dataloading.DataLoader(
                graph, idx_valid, sampler,
                batch_size=batch_size, shuffle=False, drop_last=False, use_uva=True
            )
            test_loader = _dgl.dataloading.DataLoader(
                graph, idx_test, sampler,
                batch_size=batch_size, shuffle=False, drop_last=False, use_uva=True
            )

            self.dataset.update({
                "idx_train": idx_train, "idx_valid": idx_valid, "idx_test": idx_test,
                "y_train":   y_train,   "y_valid":   y_valid,   "y_test":   y_test,
                "valid_loader": valid_loader,
                "test_loader":  test_loader,
            })

            print(
                f"[TemporalSplit] train={len(idx_train):,}  "
                f"val={len(idx_valid):,}  test={len(idx_test):,}  "
                f"(sorted by TransactionDT)"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TemporalModelHandlerModule
# ═════════════════════════════════════════════════════════════════════════════

class TemporalModelHandlerModule(ModelHandlerModule):
    """
    Extends ModelHandlerModule with:
      - Swaps DRAG for TemporalDRAG (when "apply_temporal" is true).
      - Uses a temporally-weighted CrossEntropyLoss so that recent
        transactions have a higher contribution to the training loss.

    Temporal sample weight formula
    --------------------------------
    Given a node's TransactionDT value t_i and the training set range
    [t_min, t_max], we define:

        w_i = exp(  (t_i - t_min) / (half_life * (t_max - t_min))  )

    where half_life = config["temporal_decay"] (default 0.5).
    A value of 0 disables per-sample weighting (uniform weights).
    """

    def __init__(self, configuration, datahandler: DataHandlerModule):
        super().__init__(configuration, datahandler)

        self._apply_temporal  = configuration.get("apply_temporal", True)
        self._temporal_decay  = configuration.get("temporal_decay", 0.5)
        self._time_enc_dim    = configuration.get("time_enc_dim", 16)

        if not self._apply_temporal:
            return

        # Pre-compute per-node temporal weights (CPU tensor, shape (N,))
        dt = self.dataset.get("transaction_dt", None)
        if dt is not None and self._temporal_decay > 0:
            self._node_weights = self._compute_temporal_weights(
                dt,
                self.dataset["idx_train"],
                self._temporal_decay,
            )
        else:
            self._node_weights = None

    # ── Model selection override ──────────────────────────────────────────────
    def select_model(self) -> nn.Module:
        """Replace DRAG with TemporalDRAG."""
        if not self.args.__dict__.get("apply_temporal", True):
            return super().select_model()

        import torch
        torch.cuda.empty_cache()

        graph   = self.dataset["graph"]
        feature = self.dataset["features"]

        rel_names = list(graph.etypes)   # e.g. ['card_link', 'addr_link', 'time_link']

        model = TemporalDRAG(
            feature_dim   = feature.shape[1],
            emb_dim       = self.args.emb_size,
            gat_heads     = self.args.n_head,
            num_agg_heads = self.args.n_head_agg,
            num_classes   = 2,
            is_concat     = True,
            num_relations = len(rel_names),
            feat_drop     = self.args.feat_drop,
            attn_drop     = self.args.attn_drop,
            # TemporalDRAG-specific
            time_enc_dim  = self.args.__dict__.get("time_enc_dim", 16),
            rel_names     = rel_names,
        )
        return model

    # ── Train override – adds temporal weighting to the loss ─────────────────
    def train(self):
        """
        Identical to ModelHandlerModule.train() except:
          - loss_fn is replaced with TemporalWeightedCrossEntropy when weights
            are available.
          - Everything else (sampler, optimizer, early stopping) is unchanged.
        """
        if not self._apply_temporal or self._node_weights is None:
            # Fall back to standard training
            return super().train()

        import time as _time
        import dgl as _dgl
        from utils import test, generate_batch_idx

        # ── Setup (mirrors ModelHandlerModule.train exactly) ─────────────────
        self.set_seed()
        torch.cuda.empty_cache()
        device = torch.device(self.args.cuda_id)
        torch.cuda.set_device(device)

        graph      = self.dataset["graph"]
        idx_train  = self.dataset["idx_train"]
        y_train    = self.dataset["y_train"]

        model      = self.model
        node_w_dev = self._node_weights.to(device)   # (N,)

        sampler      = _dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))
        optimizer    = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0
        recall_best = precision_best = 0.0

        print("\n", "*" * 20, " Train the DRAG (Temporal Weighting ON) ", "*" * 20)

        for epoch in range(self.epochs):
            model.train()
            avg_loss   = []
            epoch_time = 0.0
            torch.cuda.empty_cache()

            batch_idx = generate_batch_idx(idx_train, y_train, self.args.batch_size, self.args.seed)
            train_loader = _dgl.dataloading.DataLoader(
                graph, batch_idx, sampler,
                batch_size=self.args.batch_size,
                shuffle=False, drop_last=False, use_uva=True
            )

            start_time = _time.time()
            for batch in train_loader:
                _, output_nodes, blocks = batch
                blocks         = [b.to(device) for b in blocks]
                output_labels  = blocks[-1].dstdata["y"].type(torch.LongTensor).cuda()

                # Per-sample weights for this batch
                sample_weights = node_w_dev[output_nodes.to(device)]   # (B,)

                logits = model(blocks)
                loss   = _temporal_weighted_ce(logits, output_labels.squeeze(), sample_weights)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss.append(loss.item() / output_nodes.shape[0])

            epoch_time += _time.time() - start_time
            line = (f"Epoch: {epoch+1} (Best: {epoch_best}), "
                    f"loss: {np.mean(avg_loss):.6f}, time: {epoch_time:.2f}s")
            self.result.write_train_log(line, print_line=True)

            if (epoch + 1) % self.args.valid_epochs == 0:
                model.eval()
                auc_val, recall_val, f1_mac_val, precision_val = test(
                    model, self.dataset["valid_loader"],
                    self.result, epoch, epoch_best, flag="val"
                )
                gain_auc    = (auc_val    - auc_best)    / auc_best
                gain_f1_mac = (f1_mac_val - f1_mac_best) / f1_mac_best
                if (gain_auc + gain_f1_mac) > 0:
                    auc_best = auc_val; recall_best = recall_val
                    f1_mac_best = f1_mac_val; precision_best = precision_val
                    epoch_best = epoch
                    torch.save(model.state_dict(), self.result.model_path)

            if (epoch - epoch_best) > self.args.patience:
                print("\n", "*" * 20, f"Early stopping at epoch {epoch}", "*" * 20)
                break

        self.result.write_val_log(auc_best, recall_best, f1_mac_best, precision_best, epoch_best)

        print(f"Restore model from epoch {epoch_best}")
        model.load_state_dict(torch.load(self.result.model_path))

        print("\n", "*" * 20, " Test the DRAG ", "*" * 20)
        auc_test, recall_test, f1_mac_test, precision_test = test(
            model, self.dataset["test_loader"],
            self.result, epoch, epoch_best, flag="test"
        )
        return auc_test, f1_mac_test

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_temporal_weights(
        transaction_dt: np.ndarray,
        idx_train: list,
        decay: float,
    ) -> torch.Tensor:
        """
        Returns a (N,) float tensor of per-node training weights.

        Nodes outside idx_train get weight 1.0 (they only appear in
        val/test loaders where the loss is not computed).

        Formula:  w = exp( (t - t_min) / (decay * range) )
        Weights are then normalised to sum to len(idx_train) so the
        absolute scale of the loss stays comparable across runs.
        """
        N      = len(transaction_dt)
        dt     = torch.tensor(transaction_dt, dtype=torch.float32)
        weights = torch.ones(N, dtype=torch.float32)

        train_dt = dt[idx_train]
        t_min    = train_dt.min()
        t_range  = (train_dt.max() - t_min).clamp(min=1.0)

        # Exponential recency weighting
        w_train  = torch.exp((train_dt - t_min) / (decay * t_range))

        # Normalise so mean weight of training nodes = 1 (prevents loss scale shift)
        w_train  = w_train / w_train.mean()

        weights[idx_train] = w_train

        print(
            f"[TemporalWeighting] min_w={w_train.min():.3f}  "
            f"max_w={w_train.max():.3f}  mean_w={w_train.mean():.3f}  "
            f"decay={decay}"
        )
        return weights


# ─── Module-level helper (not a method so it can be JIT-compiled later) ──────

def _temporal_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample weighted cross-entropy.

    logits  : (B, C)
    labels  : (B,)  LongTensor
    weights : (B,)  FloatTensor, already on the correct device
    """
    loss_per_sample = F.cross_entropy(logits, labels, reduction="none")  # (B,)
    return (loss_per_sample * weights).mean()


# ═════════════════════════════════════════════════════════════════════════════
# PATCH INSTRUCTIONS  (what to change in ieee_cis_dataset.py)
# ═════════════════════════════════════════════════════════════════════════════
"""
In ieee_cis_dataset.py → load_ieee_cis(), find the block that builds the graph
(step 7) and add two lines to persist TransactionDT on the graph:

    BEFORE  (existing code, around line "graph.ndata['x'] = ...")
    ────────────────────────────────────────────────────────────
    graph.ndata["x"] = torch.tensor(X, dtype=torch.float32)
    graph.ndata["y"] = torch.tensor(y, dtype=torch.long)

    AFTER
    ─────
    graph.ndata["x"]              = torch.tensor(X, dtype=torch.float32)
    graph.ndata["y"]              = torch.tensor(y, dtype=torch.long)

    # ── Temporal weighting: store raw TransactionDT ──────────────
    dt_values = df["TransactionDT"].fillna(0).values.astype("float32")
    graph.ndata["transaction_dt"] = torch.tensor(dt_values, dtype=torch.float32)

    # ── Temporal weighting: attach delta_t to every edge type ────
    from post_training.temporal_drag import build_delta_t_edge_data
    graph = build_delta_t_edge_data(graph, dt_values)
    # ─────────────────────────────────────────────────────────────

Then in data_handler.py → DataHandlerModule.__init__(), the STEP-6 block
drops TransactionDT before scaling:

    feat_df = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")

This is ALREADY present in ieee_cis_dataset.py (it's dropped from features
before StandardScaler).  No change needed there — TransactionDT is stored
separately on graph.ndata, not in the feature matrix.
"""
