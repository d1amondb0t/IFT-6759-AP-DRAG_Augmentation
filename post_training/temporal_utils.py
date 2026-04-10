import torch
import dgl
import numpy as np

def build_delta_t_edge_data(graph: dgl.DGLGraph, transaction_dt: np.ndarray) -> dgl.DGLGraph:
    """
    Attach a 'delta_t' edge feature to every edge type in the DGL heterograph.

    delta_t[e] = |TransactionDT[src(e)] - TransactionDT[dst(e)]|

    Call this after building the graph in ieee_cis_dataset.py:

        graph = build_delta_t_edge_data(graph, df["TransactionDT"].values)

    Parameters
    ----------
    graph          : DGL heterograph (output of load_ieee_cis).
    transaction_dt : (N,) array of raw TransactionDT seconds for each node.

    Returns
    -------
    The same graph object with edata['delta_t'] set per edge type.
    """
    dt_tensor = torch.tensor(transaction_dt, dtype=torch.float32)

    for etype in graph.canonical_etypes:
        src_ids, dst_ids = graph.edges(etype=etype)
        delta_t = (dt_tensor[src_ids] - dt_tensor[dst_ids]).abs()
        graph.edges[etype].data["delta_t"] = delta_t

    return graph