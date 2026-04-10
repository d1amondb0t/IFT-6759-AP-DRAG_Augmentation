import pandas as pd
import torch
import dgl
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset
from ieee_cis_dataset import load_ieee_cis

DATA_NAMES = ['yelp', 'amazon', 'amazon_new', 'ieee_cis']

def load_data(data_name,
              multi_relation,
              raw_dir='./data',
              gan=False,
              graph_gan=False,
              smote=False,
              graph_smote=False,
              contrastive_learning=False,
              time_weight_decay=False):
	
	assert data_name in DATA_NAMES

	if data_name == 'yelp':
		graph = FraudYelpDataset(raw_dir).graph
	elif data_name.startswith('amazon'):
		graph = FraudAmazonDataset(raw_dir).graph
		if data_name == 'amazon_new':
			features = graph.ndata['feature'].numpy()
			mask_dup = torch.BoolTensor(pd.DataFrame(features).duplicated(keep=False).values)
			graph = graph.subgraph(~mask_dup)
	
	elif data_name == 'ieee_cis':
		return load_ieee_cis(raw_dir, apply_gan=gan,
                       apply_graph_gan=graph_gan,
                       apply_smote=smote,
                       apply_graph_smote=graph_smote,
                       apply_contrastive_learning=contrastive_learning,
                       apply_time_weight_decay=time_weight_decay)  
  
	# Rename ndata & Remove redundant data
	graph.ndata['x'] = graph.ndata['feature']
	graph.ndata['y'] = graph.ndata['label']
	del graph.ndata['feature'], graph.ndata['label']
	del graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask']
	if not multi_relation:
		graph = dgl.to_homogeneous(graph, ndata=['x', 'y'], store_type=False)
		graph.ndata['x'] = graph.ndata['feature']
		graph.ndata['y'] = graph.ndata['label']
		del graph.ndata['_ID'], graph.edata['_ID']

	for etype in graph.etypes:
		graph = dgl.add_self_loop(graph, etype=etype)

	return graph