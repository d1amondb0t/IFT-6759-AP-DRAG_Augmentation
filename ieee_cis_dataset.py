import dgl
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from feature_engineering import run_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

MAX_GROUP         = 30
MAX_EDGES_PER_REL = 200_000

def _bucketize(series):
  s = series.fillna("__nan__")
  if pd.api.types.is_numeric_dtype(s):
    try:
      return pd.qcut(s.astype(float), q=200,
                    labels=False, duplicates="drop").fillna(0).astype(int).values
    except Exception:
      pass
  return pd.Categorical(s.astype(str)).codes.astype(int)

def _make_edges(buckets):
  bmap = defaultdict(list)
  for i, b in enumerate(buckets):
    bmap[int(b)].append(i)
  srcs, dsts = [], []
  for grp in bmap.values():
    if len(grp) < 2:
      continue
    g = grp[:MAX_GROUP]
    for i in range(len(g)):
      for j in range(i + 1, len(g)):
        srcs += [g[i], g[j]]
        dsts += [g[j], g[i]]
        if len(srcs) >= MAX_EDGES_PER_REL:
          return torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)
  return torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)

def load_ieee_cis(raw_dir, sample=500000, seed=42):
  print(f"[IEEE-CIS] Loading from {raw_dir} ...")

  tr_t = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"))
  tr_i = pd.read_csv(os.path.join(raw_dir, "train_identity.csv"))
  df   = tr_t.merge(tr_i, on="TransactionID", how="left")
  print(f"  Merged:    {df.shape}  |  Fraud rate: {df['isFraud'].mean():.4f}")

  # Stratified subsample
  if 0 < sample < len(df):
      fraud   = df[df["isFraud"] == 1]
      legit   = df[df["isFraud"] == 0]
      n_fraud = max(20, int(sample * len(fraud) / len(df)))
      n_legit = sample - n_fraud
      df = pd.concat([
          fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed),
          legit.sample(n=min(n_legit, len(legit)), random_state=seed),
      ]).sample(frac=1, random_state=seed).reset_index(drop=True)
      print(f"  Sampled:   {len(df)} rows  (fraud={n_fraud}, legit={n_legit})")

  y = df["isFraud"].values.astype(np.int64)
  n = len(df)

  # ── Node features -- key must be "x" (matches data_handler.py) ────────
  feat_df = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").copy()
  feat_df = feat_df.dropna(axis=1, thresh=int(0.4 * n))
  for c in feat_df.select_dtypes(include="object").columns:
      feat_df[c] = LabelEncoder().fit_transform(
          feat_df[c].astype(str).fillna("__nan__")).astype(np.float32)
  feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
  if feat_df.shape[1] > 50:
      feat_df = feat_df[feat_df.var().nlargest(50).index]
  X = StandardScaler().fit_transform(feat_df.values.astype(np.float32))

  # ── 3 relation edge sets (mirrors YelpChi's 3 relations) ──────────────
  card_s, card_d = _make_edges(_bucketize(df["card1"] if "card1" in df.columns else pd.Series(np.arange(n) % 500)))
  addr_s, addr_d = _make_edges(_bucketize(df["addr1"] if "addr1" in df.columns else pd.Series(np.arange(n) % 300)))
  time_s, time_d = _make_edges(_bucketize(
      df["TransactionDT"] // 3600 % 24 if "TransactionDT" in df.columns
      else pd.Series(np.arange(n) % 24)))

  # ── Build heterogeneous graph (same structure as FraudYelpDataset) ─────
  graph = dgl.heterograph({
      ("transaction", "card_link", "transaction"): (card_s, card_d),
      ("transaction", "addr_link", "transaction"): (addr_s, addr_d),
      ("transaction", "time_link", "transaction"): (time_s, time_d),
  }, num_nodes_dict={"transaction": n})

  # Keys "x" and "y" — exactly what data_handler.py expects
  graph.ndata["x"] = torch.tensor(X, dtype=torch.float32)
  graph.ndata["y"] = torch.tensor(y, dtype=torch.long)

  # Self-loops on every relation (data_handler.py does this for yelp/amazon)
  for etype in graph.etypes:
      graph = dgl.add_self_loop(graph, etype=etype)

  print(f"  print:     {n:,}")
  print(f"  Etypes:    {graph.etypes}")
  print(f"  Features:  {X.shape[1]}")
  print(f"  Labels:    {np.bincount(y)}")
  return graph
  
def load_ieee_cis_modified(raw_dir, sample=250000, seed=42):
  print(f"[IEEE-CIS] Loading from {raw_dir} ...")
  
  # ── 1. Feature Engineering ──────────────
  train_df, _ = run_pipeline(
    os.path.join(raw_dir, "train_transaction.csv"),
    os.path.join(raw_dir, "train_identity.csv"),
    os.path.join(raw_dir, "test_transaction.csv"),
    os.path.join(raw_dir, "test_identity.csv"),
  )
  
  # ── 2. Stratified Subsampling ──────────────
  if 0 < sample < len(train_df):
    fraud   = train_df[train_df["isFraud"] == 1]
    legit   = train_df[train_df["isFraud"] == 0]
    n_fraud = max(20, int(sample * len(fraud) / len(train_df)))
    n_legit = sample - n_fraud
    df = pd.concat([
      fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed),
      legit.sample(n=min(n_legit, len(legit)), random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Sampled:   {len(df)} rows  (fraud={n_fraud}, legit={n_legit})")
  else:
    df = train_df
    
  n = len(df)
  y = df["isFraud"].values.astype(np.int64)
  
  # ── 3. relation edge sets (mirrors YelpChi's 3 relations) ──────────────
  card_s, card_d = _make_edges(_bucketize(df["card1"] if "card1" in df.columns else pd.Series(np.arange(n) % 500)))
  addr_s, addr_d = _make_edges(_bucketize(df["addr1"] if "addr1" in df.columns else pd.Series(np.arange(n) % 300)))
  time_s, time_d = _make_edges(_bucketize(
      df["TransactionDT"] // 3600 % 24 if "TransactionDT" in df.columns
      else pd.Series(np.arange(n) % 24)))
  
  # ── 4. Final Feature Scaling ──────────────
  feat_df = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
  feat_df = feat_df.fillna(0)
  X = StandardScaler().fit_transform(feat_df.values.astype(np.float32))
  
  # ── 5. Build Heterogenous graph ──────────────
  graph = dgl.heterograph({
    ("transaction", "card_link", "transaction"): (card_s, card_d),
    ("transaction", "addr_link", "transaction"): (addr_s, addr_d),
    ("transaction", "time_link", "transaction"): (time_s, time_d),
  }, num_nodes_dict={"transaction": n})
  
  graph.ndata["x"] = torch.tensor(X, dtype=torch.float32)
  graph.ndata["y"] = torch.tensor(y, dtype=torch.long)
  
  for etype in graph.etypes:
    graph = dgl.add_self_loop(graph, etype=etype)
  
  print(f"  Nodes:     {n:,}")
  print(f"  Etypes:    {graph.etypes}")
  print(f"  Features:  {X.shape[1]}")
  print(f"  Labels:    {np.bincount(y)}")
  
  return graph