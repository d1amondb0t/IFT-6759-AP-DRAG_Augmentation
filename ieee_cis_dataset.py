import dgl
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from feature_engineering import run_pipeline
#from post_training.temporal_drag import build_delta_t_edge_data
from post_training.temporal_utils import build_delta_t_edge_data
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from smote.IEEEFraudSmote import IEEEFraudSMOTE

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

def _apply_gan(df, target_ratio=0.05, max_fraud_for_gan=5000, seed=42, ctgan_epochs=200):
  
  # ── 1. Isolate Fraud data for training ──────────────
  fraud_df = df[df["isFraud"] == 1].copy()
  if len(fraud_df) > max_fraud_for_gan:
    fraud_df = fraud_df.sample(max_fraud_for_gan, random_state=seed)
  
  # ── 2. SDV requires data to be properly typed, hence converting objects -> strings ──────────────
  numeric_cols = fraud_df.select_dtypes(include=[np.number]).columns
  fraud_df[numeric_cols] = fraud_df[numeric_cols].fillna(fraud_df[numeric_cols].median())
  
  # Fill remaining object/categorical nulls with a placeholder string
  obj_cols = fraud_df.select_dtypes(include=['object', 'string', 'category']).columns
  fraud_df[obj_cols] = fraud_df[obj_cols].fillna("unknown")
  
  for c in obj_cols:
      fraud_df[c] = fraud_df[c].astype("string")

  metadata = Metadata.detect_from_dataframe(data=fraud_df)
  
  # ── 3. Apply Tabular GAN Augmentation ──────────────
  synthesizer = CTGANSynthesizer(metadata,
                                 epochs=ctgan_epochs,
                                 verbose=False,
                                 enforce_rounding=False)

  synthesizer.fit(fraud_df)
  
  # ── 4. Calculate num rows required to reach fraud ratio ──────────────
  fraud_count = df["isFraud"].sum()
  total_count = len(df)
  n_synth = int(np.ceil(max(0, (target_ratio * total_count - fraud_count) / (1 - target_ratio))))
  
  if n_synth > 0:
    synthetic_fraud = synthesizer.sample(num_rows=n_synth)
    synthetic_fraud["isFraud"] = 1
    
    # Ensure new TransactionIDs don't overlap
    max_id = df["TransactionID"].max() if "TransactionID" in df.columns else 0
    synthetic_fraud["TransactionID"] = np.arange(max_id + 1, max_id + 1 + len(synthetic_fraud))
    
    df = pd.concat([df, synthetic_fraud], ignore_index=True)
    
  return df.reset_index(drop=True)


def _apply_graphgan(df, emb_dim=32, graphgan_results_dir="./GraphGAN/results/link_prediction"):
  df = df.copy()
  df["gg_node_id"] = np.arange(len(df))
  
  gen_emb_path = os.path.join(graphgan_results_dir, "ieee_cis_gan_graphgan_gen_.emb")
  dis_emb_path = os.path.join(graphgan_results_dir, "ieee_cis_gan_graphgan_dis_.emb")
  
  if not os.path.exists(gen_emb_path) or not os.path.exists(dis_emb_path):
    print("  [Warning] GraphGAN embeddings not found. Skipping GraphGAN augmentation.")
    return df.drop(columns=["gg_node_id"])
  
  def read_emb(path, prefix):
    emb = pd.read_csv(path, sep="\t", header=None, skiprows=1)
    emb.rename(columns={0: "gg_node_id"}, inplace=True)
    emb.columns = ["gg_node_id"] + [f"{prefix}_{i}" for i in range(emb.shape[1] - 1)]
    return emb
  
  emb_gen = read_emb(gen_emb_path, "gg_gen")
  emb_dis = read_emb(dis_emb_path, "gg_dis")
  
  emb_merged = emb_gen.merge(emb_dis, on="gg_node_id", how="inner")
  
  actual_gen_cols = [c for c in emb_merged.columns if c.startswith("gg_gen_")]
  actual_dim = len(actual_gen_cols)
  print(f"    Detected {actual_dim} GraphGAN embedding dimensions.")
  
  avg_embs = {"gg_node_id": emb_merged["gg_node_id"].astype(int)}
  for i in range(actual_dim):
    avg_embs[f"gg_emb_{i}"] = (
      emb_merged[f"gg_gen_{i}"].astype(float) + 
      emb_merged[f"gg_dis_{i}"].astype(float)
    ) / 2.0
  
  emb_df = pd.DataFrame(avg_embs)
  
  df = df.merge(emb_df, on="gg_node_id", how="left")
  
  gg_emb_cols = [c for c in df.columns if c.startswith("gg_emb_")]
  df[gg_emb_cols] = df[gg_emb_cols].fillna(0.0)
  
  return df.drop(columns=["gg_node_id"])


def _apply_smote(df, target_ratio=0.05):
  smote = IEEEFraudSMOTE(target_ratio=target_ratio)
  df = smote.augment(df).reset_index(drop=True)
  return df
  
def load_ieee_cis(raw_dir, 
                  sample=500000, 
                  seed=42, 
                  apply_gan=False,
                  apply_graph_gan=False,
                  apply_smote=False,
                  apply_graph_smote=False,
                  apply_contrastive_learning=False,
                  apply_time_weight_decay=False
                  ):
  
  print(f"[IEEE-CIS] Loading from {raw_dir} ...... (GAN: {apply_gan}, GraphGAN: {apply_graph_gan}, SMOTE: {apply_smote}, GraphSMOTE: {apply_graph_smote}, Contrastive Learning: {apply_contrastive_learning}, Temporal Weight Decay: {apply_time_weight_decay})")
  
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
  
  # ── 3. Apply Tabular GAN Augmentation ──────────────
  if apply_smote:
    print ("  Running tabular SMOTE to augment fraud samples...")
    df = _apply_smote(train_df)
    
  # ── 4. Apply Tabular GAN Augmentation ──────────────
  if apply_gan:
    print("  Running CTGAN to augment fraud samples...")
    df = _apply_gan(df, seed=seed)
  
  # ── 5. Apply GraphGAN Feature Embeddings ──────────────
  if apply_graph_gan:
    print("  Running GraphGAN to append node embeddings...")
    df = _apply_graphgan(df)
    
  n = len(df)
  y = df["isFraud"].values.astype(np.int64)
  
  # ── 6. relation edge sets (mirrors YelpChi's 3 relations) ──────────────
  card_s, card_d = _make_edges(_bucketize(df["card1"] if "card1" in df.columns else pd.Series(np.arange(n) % 500)))
  addr_s, addr_d = _make_edges(_bucketize(df["addr1"] if "addr1" in df.columns else pd.Series(np.arange(n) % 300)))
  time_s, time_d = _make_edges(_bucketize(
      df["TransactionDT"] // 3600 % 24 if "TransactionDT" in df.columns
      else pd.Series(np.arange(n) % 24)))
  
  # ── 7. Final Feature Scaling ──────────────
  feat_df = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
  
  for c in feat_df.select_dtypes(include=["object", "string", "category"]).columns:
      feat_df[c] = LabelEncoder().fit_transform(
          feat_df[c].astype(str).fillna("__nan__")
      ).astype(np.float32)
  
  feat_df = feat_df.fillna(0)
  X = StandardScaler().fit_transform(feat_df.values.astype(np.float32))
  
  # ── 8. Build Heterogenous graph ──────────────
  graph = dgl.heterograph({
    ("transaction", "card_link", "transaction"): (card_s, card_d),
    ("transaction", "addr_link", "transaction"): (addr_s, addr_d),
    ("transaction", "time_link", "transaction"): (time_s, time_d),
  }, num_nodes_dict={"transaction": n})
  
  graph.ndata["x"] = torch.tensor(X, dtype=torch.float32)
  graph.ndata["y"] = torch.tensor(y, dtype=torch.long)
  
  if apply_time_weight_decay:
    # ── 9. Temporal weighting: store raw TransactionDT on the graph ──────────────
    dt_values = df["TransactionDT"].fillna(0).values.astype("float32")
    graph.ndata["transaction_dt"] = torch.tensor(dt_values, dtype=torch.float32)

    # ── Temporal weighting: attach |dt_src - dt_dst| to every edge type ──
    graph = build_delta_t_edge_data(graph, dt_values)
    # ────────────────────────────────────────────────────────────────────────────
  
  for etype in graph.etypes:
    graph = dgl.add_self_loop(graph, etype=etype)
  
  print(f"  Nodes:     {n:,}")
  print(f"  Etypes:    {graph.etypes}")
  print(f"  Features:  {X.shape[1]}")
  print(f"  Labels:    {np.bincount(y)}")
  
  return graph