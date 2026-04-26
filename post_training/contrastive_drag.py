"""
contrastive_drag.py
====================
Contrastive fine-tuning pipeline for a pre-trained DRAG model.

Pipeline
--------
Phase 1  (done externally) : Train DRAG normally on labelled data.
Phase 2  (this file)       : Contrastive fine-tuning via NT-Xent loss.
Phase 3A (this file)       : Direct inference with tuned encoder.
Phase 3B (this file)       : Supervised fine-tuning on top of tuned encoder.

Usage
-----
		from contrastive_drag import run_contrastive_pipeline

		# drag_model  : already-trained DRAG model (nn.Module)
		# graph       : DGL heterograph with ndata['x'] and ndata['y']
		# config      : your existing ieee_cis.json config dict
		run_contrastive_pipeline(drag_model, graph, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.dataloading
import numpy as np
from sklearn.metrics import (
		roc_auc_score, classification_report,
		f1_score, recall_score, precision_score,
		accuracy_score, average_precision_score
)

def _print_metrics(labels, preds, probs, header, epoch_best=None):
		
		auc      = roc_auc_score(labels, probs)
		recall   = recall_score(labels, preds, pos_label=1, zero_division=0)
		f1       = f1_score(labels, preds, pos_label=1, zero_division=0)
		f1_mac   = f1_score(labels, preds, average='macro', zero_division=0)
		rec_mac  = recall_score(labels, preds, average='macro', zero_division=0)
		prec     = precision_score(labels, preds, pos_label=1, zero_division=0)
		acc      = accuracy_score(labels, preds)
		ap       = average_precision_score(labels, probs)

		epoch_str = f"- Epoch_Best: {epoch_best}\t" if epoch_best is not None else ""

		if header == 'validation':
				print(f"\nValidation performance: {epoch_str}\n"
							f"- AUC-ROC: {auc:.4f}\t"
							f"- Recall: {recall:.4f}\t"
							f"- F1-macro: {f1_mac:.4f}\t"
							f"- Precision: {prec:.4f}")
		else:
				tag = f" [{header}]" if header not in ('test', 'validation') else ""
				print(f"\n ********************  Test the DRAG{tag}  ********************")
				print(f"Test performance: {epoch_str}"
							f"- F1: {f1:.4f}\t"
							f"- Recall: {recall:.4f}\t"
							f"- Precision: {prec:.4f}\t"
							f"- Accuracy: {acc:.4f}\t"
							f"- AUC-ROC: {auc:.4f}\t"
							f"- F1-macro: {f1_mac:.4f}\t"
							f"- Recall-macro: {rec_mac:.4f}\t"
							f"- AP: {ap:.4f}")

		return auc, recall, f1, f1_mac, prec, acc, ap

# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAPH AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def drop_edges(graph, drop_ratio=0.2):
		"""
		Randomly remove a subset of edges from every relation type,
		then re-add self-loops so no node ends up with zero in-degree.
		"""
		new_data = {}
		
		#canonical_etypes returns: ('item', 'relation_type', 'item')
		for etype in graph.canonical_etypes:
				src, dst = graph.edges(etype=etype[1])
				keep = torch.rand(src.shape[0]) > drop_ratio
				new_data[etype] = (src[keep], dst[keep])

		# Create augmented graph representation
		aug = dgl.heterograph(
				new_data,
				num_nodes_dict={nt: graph.num_nodes(nt) for nt in graph.ntypes}
		)
		
		# Copy all node data (features, labels)
		for ntype in graph.ntypes:
				for k, v in graph.nodes[ntype].data.items():
						aug.nodes[ntype].data[k] = v.clone()

		# Self-loops guarantee no zero-in-degree nodes after dropping
		for etype in aug.etypes:
				aug = dgl.add_self_loop(aug, etype=etype)

		return aug


def mask_node_features(graph, node_type, mask_ratio=0.2):
		"""
		Randomly zero out mask_ratio fraction of feature dimensions
		for every node.  Node count / topology is untouched.
		"""
		
		aug = graph.clone()
		feat = aug.nodes[node_type].data['x'].clone()
		
		# if mask condition is met, we set the feat values to 0
		feat[torch.rand_like(feat) < mask_ratio] = 0.0
		aug.nodes[node_type].data['x'] = feat
		return aug


def augment_graph(graph, node_type='transaction',
									edge_drop=0.2, feat_mask=0.2):
		"""
		Produce two asymmetric views of the same graph.

		View 1 — heavier edge drop, lighter feature mask - learn structural representation

		View 2 — lighter edge drop, heavier feature mask - learn feature representation

		Asymmetry follows GRACE / GraphCL - different augmentations strengthen encoder representation.
		"""
		view1 = mask_node_features(
				drop_edges(graph, drop_ratio=edge_drop),  # heavy edge drop
				node_type=node_type,
				mask_ratio=feat_mask * 0.5                # light feature mask
		)
		view2 = drop_edges(
				mask_node_features(graph, 
													 node_type=node_type,
													 mask_ratio=feat_mask), # heavy feature mask
				drop_ratio=edge_drop * 0.5                # light edge drop
		)
		return view1, view2


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROJECTION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
		"""
		Two-layer MLP that maps encoder embeddings into the contrastive space.

		NT-Xent loss is computed HERE, not on the raw encoder output.  
		After contrastive training this head is discarded
		"""
		
		def __init__(self, in_dim, hidden_dim=128, out_dim=64):
				super().__init__()
				self.proj_head = nn.Sequential(
						nn.Linear(in_dim, hidden_dim),
						nn.ReLU(),
						nn.Linear(hidden_dim, out_dim)
				)

		def forward(self, x):
				return self.proj_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NT-XENT LOSS
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=0.5):
		"""
		NT-Xent (Normalised Temperature-scaled Cross Entropy) loss.

		For a batch of N nodes:
			- z1[i] and z2[i] are the positive pair  (same node, different views)
			- all other z1/z2 combinations are negatives

		Steps
		-----
		1. L2-normalise both sets of embeddings so cosine similarity
			 reduces to a dot product (magnitude is ignored).
		2. Concatenate into a [2N, D] matrix.
		3. Compute the full [2N, 2N] similarity matrix, scaled by temperature.
		4. Mask the diagonal (a vector is trivially similar to itself).
		5. For each row i, the positive is at position i+N (or i-N).
			 Cross-entropy over this labelling is the contrastive loss.

		Args
		----
		z1, z2      : [N, D] projected embeddings from view 1 and view 2
		temperature : scalar, lower = sharper distribution (typical: 0.1–0.5)

		Returns
		-------
		scalar loss
		"""
		N = z1.shape[0]

		# Step 1 — L2 normalise
		z1 = F.normalize(z1, dim=1)
		z2 = F.normalize(z2, dim=1)

		# Step 2 — Concatenate: [2N, D]
		z = torch.cat([z1, z2], dim=0)

		# Step 3 — Similarity matrix: [2N, 2N]
		sim = torch.mm(z, z.T) / temperature

		# Step 4 — Mask self-similarity on the diagonal
		mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
		sim = sim.masked_fill(mask, -1e9)

		# Step 5 — Positive pair labels:
		#   row i   (from z1) → positive is at position i+N (in z2)
		#   row i+N (from z2) → positive is at position i   (in z1)
		labels = torch.cat([
				torch.arange(N, 2 * N, device=z.device),
				torch.arange(0,     N, device=z.device)
		])

		return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRACT INTERMEDIATE EMBEDDINGS FROM DRAG
# ─────────────────────────────────────────────────────────────────────────────

def get_intermediate_embeddings(drag_model, graph, n_layers,
																 batch_size, device, with_grad=False):
		"""
		Run DRAG's forward pass over the full graph to intercept embeddings

		DRAG outputs [N, 2] logits are too small, requires the embedding [N, emb_size[-1]]

		How to intercept embeddings:
		1) Temporarily replace DRAG classification head with nn.Identity() to return raw embeddings
		2) Restore original head

		Returns
		-------
		embeddings : [N, emb_size[-1]] tensor, ordered by node index
		"""
		# 1 - Swap out the classifier head
		original_classifier = drag_model.linear_layer
		drag_model.linear_layer = nn.Identity()

		# 2 - Build neighbor sampler and DataLoader 
		all_idx = torch.arange(graph.num_nodes(), dtype=torch.long, device=graph.device)
		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
		loader  = dgl.dataloading.DataLoader(
				graph, all_idx, sampler,
				batch_size=batch_size, shuffle=False,
				drop_last=False, use_uva=False
		)

		parts = []   # embeddings in batch order
		nids  = []   # corresponding node indices

		context = torch.enable_grad() if with_grad else torch.no_grad()
		with context:
				for input_nodes, output_nodes, blocks in loader:
						blocks = [b.to(device) for b in blocks]
						# DRAG's forward: forward(blocks, attn_coeff=None)
						out = drag_model(blocks, attn_coeff=None)   # [batch, emb_dim]
						parts.append(out)
						nids.append(output_nodes.to(device))

		all_embs = torch.cat(parts, dim=0)   # [N, emb_dim] (batch order)
		all_nids = torch.cat(nids,  dim=0)   # [N]          (batch order)

		# 3 - Reorder to canonical node order
		# Use index_put_ instead of fancy indexing to preserve the grad graph
		N      = graph.num_nodes()
		D      = all_embs.shape[1]
		ordered = torch.zeros(N, D, device=device,
													requires_grad=with_grad)
		idx_exp = all_nids.unsqueeze(1).expand(-1, D)
		ordered = ordered.scatter(0, idx_exp, all_embs)

		# 4 - Restore the original classifier head
		drag_model.linear_layer = original_classifier

		return ordered


# ─────────────────────────────────────────────────────────────────────────────
# 5. PHASE 2 — CONTRASTIVE FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def train_contrastive(drag_model, graph, config,
											node_type='transaction',
											proj_hidden=128, proj_out=64,
											temperature=0.5,
											edge_drop=0.2, feat_mask=0.2,
											epochs=50, lr=1e-3,
											device=None):
		"""
		Phase 2: contrastively fine-tune a pre-trained DRAG model.

		Each epoch:
			1. Produce two augmented views of the graph.
			2. Extract intermediate embeddings from DRAG for both views,
				 keeping gradients so the loss can update DRAG's weights.
			3. Project both embeddings through the projection head.
			4. Compute NT-Xent loss and backpropagate.

		Returns
		-------
		drag_model : DRAG with updated encoder weights (classifier head intact)
		"""
		if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		n_layers   = len(config['emb_size'])
		batch_size = config['batch_size']

		drag_model = drag_model.to(device)
		graph      = graph.to(device)
		emb_dim = config['emb_size'][-1]
		
		print(f"[Contrastive] Encoder embedding dim: {emb_dim}")

		# 1 - Build projection head
		proj_head = ProjectionHead(emb_dim, proj_hidden, proj_out).to(device)

		# Optimize both encoder and projection head together
		optimizer = optim.Adam(
				list(drag_model.parameters()) + list(proj_head.parameters()),
				lr=lr, weight_decay=1e-5
		)

		# ── Training loop ─────────────────────────────────────────────────────
		drag_model.train()
		proj_head.train()

		for epoch in range(1, epochs + 1):
				# Step 1 — augment
				view1, view2 = augment_graph(graph, node_type, edge_drop, feat_mask)
				view1 = view1.to(device)
				view2 = view2.to(device)

				# Step 2 — encode both views (gradients kept)
				h1 = get_intermediate_embeddings(
						drag_model, view1, n_layers, batch_size, device, with_grad=True
				)
				h2 = get_intermediate_embeddings(
						drag_model, view2, n_layers, batch_size, device, with_grad=True
				)

				# Step 3 — project
				z1 = proj_head(h1)
				z2 = proj_head(h2)

				# Step 4 — loss + backprop
				loss = nt_xent_loss(z1, z2, temperature)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				#if epoch % 10 == 0:
				print(f"  Epoch {epoch:3d}/{epochs}  Loss: {loss.item():.4f}")

		# Projection head discarded
		print("[Contrastive] Fine-tuning complete. Projection head discarded.")
		drag_model.eval()
		return drag_model

def train_contrastive_fixed(drag_model, graph, config,
											node_type='transaction',
											proj_hidden=128, proj_out=64,
											temperature=0.5,
											edge_drop=0.2, feat_mask=0.2,
											epochs=50, lr=1e-3,
											mini_batch_size= 4096,
											device=None):
		"""
		Phase 2: contrastively fine-tune a pre-trained DRAG model.

		Each epoch:
			1. Temporarily replace DRAG classification head with nn.Identity() to return raw embeddings
			2. Produce two augmented views of the graph.
			3. Extract intermediate embeddings from DRAG for both views,
				 keeping gradients so the loss can update DRAG's weights.
			4. Project both embeddings through the projection head.
			5. Compute NT-Xent loss and backpropagate.
			6. Restore original head

		Returns
		-------
		drag_model : DRAG with updated encoder weights (classifier head intact)
		"""
		if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# 1 - Swap out the classifier head
		original_classifier = drag_model.linear_layer
		drag_model.linear_layer = nn.Identity()    

		n_layers   = len(config['emb_size'])
		batch_size = mini_batch_size if mini_batch_size else config['batch_size']
		print(f"[Contrastive] Batch Size: {batch_size}")

		drag_model = drag_model.to(device)
		graph      = graph.to(device)
		emb_dim    = config['emb_size'][-1]
		
		print(f"[Contrastive] Encoder embedding dim: {emb_dim}")

		# Build projection head
		proj_head = ProjectionHead(emb_dim, proj_hidden, proj_out).to(device)

		# Optimize both encoder and projection head together
		optimizer = optim.Adam(
				list(drag_model.parameters()) + list(proj_head.parameters()),
				lr=lr, weight_decay=1e-5
		)

		# Training Loop
		drag_model.train()
		proj_head.train()

		total_nodes = graph.num_nodes(node_type)

		for epoch in range(1, epochs + 1):
				# Step 2 — augment 2 graph views at each epoch
				view1, view2 = augment_graph(graph, node_type, edge_drop, feat_mask)
				view1 = view1.to(device)
				view2 = view2.to(device)

				perm = torch.randperm(total_nodes, device=device)
				sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

				train_loader = dgl.dataloading.DataLoader(
						view1,
						torch.arange(total_nodes).to(device),
						sampler,
						batch_size=batch_size,
						shuffle=True,
						drop_last=False
				)

				num_batches = (total_nodes + batch_size -1) // batch_size
				epoch_loss = 0.0

				for (inputs, output_nodes, blocks1) in train_loader:
						blocks1 = [b.to(device) for b in blocks1]

						_, _, blocks2 = sampler.sample(view2, output_nodes)
						blocks2 = [b.to(device) for b in blocks2]


						# Step 3 — encode both views (gradients kept)
						h1 = drag_model(blocks1, attn_coeff=None)
						h2 = drag_model(blocks2, attn_coeff=None)

						# Step 4 -Project embeddings in different linear space
						z1 = proj_head(h1)
						z2 = proj_head(h2)

						# Step 5 — loss + backprop
						loss = nt_xent_loss(z1, z2, temperature)

						optimizer.zero_grad()
						loss.backward()
						optimizer.step()

						epoch_loss += loss.item()

						del h1, h2, z1, z2, blocks1, blocks2, loss
				
				del view1, view2

				avg_loss = epoch_loss / num_batches
				print(f"  Epoch {epoch:3d}/{epochs}  Loss: {avg_loss:.4f}")

		# Step 6 - Restore Classifier Head & Discard Trained Projection Head
		drag_model.linear_layer = original_classifier
		
		del proj_head, optimizer
		print("[Contrastive] Fine-tuning complete. Projection head discarded.")
		drag_model.eval()
		return drag_model

# ─────────────────────────────────────────────────────────────────────────────
# 6. PHASE 3B — SUPERVISED FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def finetune_supervised(drag_model, graph, config,
												node_type='transaction',
												epochs=30, lr=1e-3, device=None):
		"""
		Phase 3B: re-run supervised training on top of the contrastively-tuned
		encoder.
		
		Returns
		-------
		drag_model : fully fine-tuned DRAG model
		"""
		
		if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		n_layers   = len(config['emb_size'])
		batch_size = config['batch_size']

		drag_model = drag_model.to(device)
		graph      = graph.to(device)

		labels = graph.nodes[node_type].data['y'].cpu()
		N      = graph.num_nodes(node_type)

		# 1 - Train/val split
		n_train  = int(N * config['train_ratio'])
		all_idx  = torch.randperm(N, device=graph.device)
		train_idx = all_idx[:n_train]
		val_idx   = all_idx[n_train:]

		sampler       = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
		train_loader  = dgl.dataloading.DataLoader(
				graph, train_idx, sampler,
				batch_size=batch_size, shuffle=True,
				drop_last=False, use_uva=False
		)

		# 2 - Optimiser
		# To fine-tune encoder + head: use all parameters (default below)
		# To freeze encoder and only train head, replace with:
		#   optimizer = optim.Adam(drag_model.classifier.parameters(), lr=lr)
		optimizer = optim.Adam(drag_model.parameters(), lr=lr, weight_decay=1e-5)
		criterion = nn.CrossEntropyLoss()

		# 3 - Training loop
		for epoch in range(1, epochs + 1):
				drag_model.train()
				total_loss = 0.0
				n_batches  = 0

				for input_nodes, output_nodes, blocks in train_loader:
						blocks  = [b.to(device) for b in blocks]
						logits  = drag_model(blocks, attn_coeff=None)    # [batch, 2]
						targets = labels[output_nodes.cpu()].to(device)

						loss = criterion(logits, targets)
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()

						total_loss += loss.item()
						n_batches  += 1

				if epoch % 5 == 0:
						avg_loss = total_loss / max(n_batches, 1)
						print(f"  [Phase 3B] Epoch {epoch:3d}/{epochs}  "
									f"Loss: {avg_loss:.4f}")

		# 4 - Final evaluation on validation set
		# Collect logits and labels in batch order
		drag_model.eval()
		val_loader = dgl.dataloading.DataLoader(
				graph, val_idx, sampler,
				batch_size=batch_size, shuffle=False,
				drop_last=False, use_uva=False
		)

		all_logits, all_labels = [], []
		with torch.no_grad():
				for input_nodes, output_nodes, blocks in val_loader:
						blocks = [b.to(device) for b in blocks]
						logits = drag_model(blocks, attn_coeff=None)   # [batch, 2]
						all_logits.append(logits.cpu())
						all_labels.append(labels[output_nodes.cpu()])  # aligned labels

		all_logits = torch.cat(all_logits, dim=0)   # [N_val, 2]
		all_labels = torch.cat(all_labels, dim=0)   # [N_val]

		val_probs  = torch.softmax(all_logits, dim=1)[:, 1].numpy()
		val_preds  = all_logits.argmax(dim=1).numpy()
		val_labels = all_labels.numpy()

		_print_metrics(val_labels, val_preds, val_probs, header='validation')

		return drag_model

# ─────────────────────────────────────────────────────────────────────────────
# 7. TEST SET EVALUATION (mirrors original DRAG test logic)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(drag_model, data_handler, config,
										 node_type='transaction', device=None, label=''):
		"""
		Evaluate drag_model on the exact same test split that data_handler
		created during the original DRAG training, so results are directly
		comparable to the baseline.
		"""
		if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		graph      = data_handler.dataset['graph'].to(device)
		idx_test   = data_handler.dataset['idx_test']
		node_labels = data_handler.dataset['labels'].cpu()
		n_layers   = len(config['emb_size'])
		batch_size = config['batch_size']

		sampler     = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
		test_loader = dgl.dataloading.DataLoader(
				graph,
				torch.tensor(idx_test, dtype=torch.long, device=graph.device)
						if not isinstance(idx_test, torch.Tensor)
						else idx_test.to(graph.device),
				sampler,
				batch_size=batch_size, shuffle=False,
				drop_last=False, use_uva=False
		)

		drag_model.eval()
		all_logits, all_labels = [], []
		with torch.no_grad():
				for input_nodes, output_nodes, blocks in test_loader:
						blocks = [b.to(device) for b in blocks]
						logits = drag_model(blocks, attn_coeff=None)
						all_logits.append(logits.cpu())
						all_labels.append(node_labels[output_nodes.cpu()])

		all_logits = torch.cat(all_logits, dim=0)
		all_labels = torch.cat(all_labels, dim=0)

		probs  = torch.softmax(all_logits, dim=1)[:, 1].numpy()
		preds  = all_logits.argmax(dim=1).numpy()
		labels = all_labels.numpy()

		auc, *_ = _print_metrics(labels, preds, probs,
															header=label if label else 'test')
		return auc, preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# 8. CONTRASTIVE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_contrastive_pipeline(drag_model, graph, config,
															data_handler=None,
															node_type='transaction',
															run_3a=True, run_3b=True):
		"""
		Run the full contrastive learning pipeline end-to-end.
		"""
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(f"Device: {device}\n")

		# Phase 2: Contrastive fine-tuning 
		print("=" * 60)
		print("PHASE 2 — Contrastive Fine-tuning")
		print("=" * 60)
		drag_model = train_contrastive_fixed(
				drag_model=drag_model,
				graph=graph,
				config=config,
				node_type=node_type,
				device=device
		)

		# Phase 3A: Direct inference — no further training
		if run_3a:
				print("\n" + "=" * 60)
				print("PHASE 3A — Direct Inference (contrastive learning, no retraining)")
				print("=" * 60)
				if data_handler is not None:
						evaluate_on_test(drag_model, data_handler, config,
														 node_type=node_type, device=device,
														 label='Phase 3A — Contrastive encoder')

		# Phase 3B: Supervised fine-tuning on top of contrastive encoder
		if run_3b:
				print("\n" + "=" * 60)
				print("PHASE 3B — Supervised Fine-tuning on Contrastive Encoder")
				print("=" * 60)
				drag_model = finetune_supervised(
						drag_model=drag_model,
						graph=graph,
						config=config,
						node_type=node_type,
						device=device
				)
				if data_handler is not None:
						evaluate_on_test(drag_model, data_handler, config,
														 node_type=node_type, device=device,
														 label='Phase 3B — Supervised fine-tune')

		return drag_model
