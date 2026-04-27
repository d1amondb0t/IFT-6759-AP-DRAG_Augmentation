# Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection
This code is the official implementation of the following [paper](https://ieeexplore.ieee.org/document/10411688):

> Heehyeon Kim, Jinhyeok Choi, and Joyce Jiyoung Whang, Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection, Machine Learning on Graphs (MLoG) Workshop at the 23rd IEEE International Conference on Data Mining (ICDM), 2023

All codes are written by Heehyeon Kim (heehyeon@kaist.ac.kr) and Jinhyeok Choi (cjh0507@kaist.ac.kr). When you use this code, please cite our paper.

```bibtex
@article{drag,
  author={Heehyeon Kim, Jinhyeok Choi, and Joyce Jiyoung Whang},
  booktitle = {2023 IEEE International Conference on Data Mining Workshops (ICDMW)},
  title = {Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection},
  year = {2023},
  pages = {1092-1096}
}
```

## Requirments
We used Python 3.8, Pytorch 1.12.1, and DGL 1.0.2 with cudatoolkit 11.3.

## Usage
### DRAG
We used NVIDIA RTX A6000 and NVIDIA GeForce RTX 3090 for all our experiments. We provide the template configuration file (`template.json`) for the datasets.

## How to run the models

There are two ways to run the project.

For a single DRAG experiment using a JSON configuration file, run:

```bash
python run.py --exp_config_path=./template.json
```

This runs one experiment using the settings defined in template.json.

For the updated augmented pipeline, use main.ipynb. This notebook runs the full set of augmented experiments by combining different preprocessing and post-training options, including GAN, GraphGAN, SMOTE, GraphSMOTE, contrastive learning, and time-weight decay.

Results will be printed in the terminal and saved in the directory designated by the configuration file.

Each run corresponds to an experiment ID `f"{dataset_name}-{train_ratio}-{seed}-{time}"`.

You can find log files and pandas DataFrame pickle files associated with experiment IDs in the designated directory.

There are some useful functions to handle experiment results in `utils.py`.

You can find an example in `performance_check.ipynb`.

### Training from Scratch
To train DRAG from scratch, run `run.py` with the configuration file. Please refer to `model_handler.py`, `data_handler.py`, and `model.py` for examples of the arguments in the configuration file.

The list of arguments of the configuration file:
- `--seed`: seed
- `--data_name`: name of the fraud detection dataset (available datasets are YelpChi(`yelp`) and Amazon_new(`amazon_new`))
- `--raw_dir`: the directory of the datasets
- `--sample`: the number of elements to sample from the original dataset for training
- `--multi_relation`: whether to use the original multi-relation graph structure. If set to False, the graph is converted to a homogeneous graph
- `--n_head`: a list consisting of the number of heads for each DRAGConv layer $N_{\alpha}$
- `--n_head_agg`: a list consisting of the number of heads for aggregation from different relations $N_{\gamma}$ and layers $N_{\beta}$
- `--train_ratio`: train ratio
- `--test_ratio`: test ratio
- `--emb_size`: a list consisting of the embedding size $d'$ for each DRAGConv layer
- `--lr`: learning rate
- `--weight_decay`: weight decay
- `--feat_drop`: feature dropout rate for DRAGConv layer
- `--attn_drop`: attention dropout rate for DRAGConv layer
- `--epochs`: total number of training epochs 
- `--valid_epochs`: the duration of validation
- `--batch_size`: the batch size
- `--patience`: early stopping patience
- `cuda_id`: CUDA device ID used for training
- `--save_dir`: directory path for saving train, validation, test logs, and the best model
- `apply_gan`: whether to apply tabular GAN-based augmentation for the IEEE-CIS dataset
- `apply_graph_gan`: whether to apply GraphGAN embedding features
- `apply_smote`: whether to apply SMOTE-based minority-class oversampling
- `apply_graph_smote`: whether to apply graph-level SMOTE augmentation during IEEE-CIS data loading
- `use_embedding_smote`: whether to apply the advanced embedding-based GraphSMOTE step after the train/validation/test split
- `apply_contrastive_learning`: whether to apply contrastive-learning-based representation processing
- `apply_time_weight_decay`: whether to apply time-decay weighting information to transaction graph edges

## Hyperparameters
We tuned DRAG with the following tuning ranges:
- `lr`: {0.01, 0.001}
- `weight_decay`: {0.001, 0.0001}
- `feat_drop`: 0.0
- `attn_drop`: 0.0
- `len(emb_size)`: $L$ = {1, 2, 3}
- `n_head`: $N_{\alpha}$ = {2, 8}
- `n_head_agg`: $N_{\beta}$ , $N_{\gamma}$ = {2, 8}
- `train_ratio`: {0.01, 0.1, 0.4}
- `test_ratio`: 0.67
- `batch_size`: 1024
- `embedding size`: $d'$ = 64
- `epochs`: 1000
- `patience`: 100

## Description for each file
- `datasets.py`: A file for loading the YelpChi and Amazon_new datasets
- `data_handler.py`: A file for processing the given dataset according to the arguments
- `feature_engineering.py`: A file for performing feature engineering
- `ieee_cis_dataset.py`: A file for loading and preprocessing the IEEE-CIS fraud-detection dataset, applying optional CTGAN, SMOTE, GraphGAN, contrastive-learning, and temporal-weight-decay augmentations, and constructing the heterogeneous transaction graph
- `layers.py`: A file for defining the DRAGConv layer
- `main.ipynb`: A notebook for running or experimenting with the augmented DRAG pipeline interactively
- `model_handler.py`: A file for training DRAG
- `models.py`: A file for defining DRAG architecture
- `performance_check.ipynb`: A file for checking the fraud detection performance of DRAG
- `run.py`: A file for running DRAG on the YelpChi and Amazon_new datasets
- `result_manager.py`: A file for managing train, validation, and test logs
- `template.json`: A template file consisting of arguments
- `utils.py`: A file for defining utility functions
- `GraphGAN/results/link_prediction/`: A folder for storing GraphGAN link-prediction outputs, including the generated graph input file and the learned generator/discriminator embedding files used by the GraphGAN augmentation pipeline.
- `experiment_results/`: A folder for storing saved experiment results from different augmentation settings and repeated runs. The result files are organized by pre-training augmentation method, post-training method, and run number, with a `summary.csv` file for aggregated results.
- `post_training/`: A folder containing optional post-training modules applied after the base DRAG model is trained, including contrastive learning, temporal DRAG utilities, and time-weighted training components.
- `smote/`: A folder containing the SMOTE-based augmentation implementation for the IEEE-CIS fraud detection dataset.
