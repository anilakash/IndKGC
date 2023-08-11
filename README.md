# IndKGC: Inductive Knowledge Graph Completion with GNNs and Rules:An Analysis #

This is the official implemetation of the paper Inductive Knowledge Graph Completion with GNNs and Rules:An Analysis.

Authors:
[Akash Anil, Victor Gutierrez Basulto, Yazmín Ibáñez-García, Steven Schockaert]

## Overview ##
IndKGC presents an analysis over Knowledge Graph Completion in inductive setting where Training and Testing knowledge graphs are disjoint in terms of entities. This setting forces the models to predict the future connections or links only using the graph strcuture (relation, induced graphs, etc.) not the entities. Looking at the inductive scenario, rule-based link prediction seems to be a natural choice. However, the state-of-the-art methods are only based on the GNN frameworks. This paper first revisits the rule-based method namely, AnyBURL and evaluates the link prediction in inductive setting. We find that the rule-based methods are unfairly disadvataged by the current evaluation setting. Further, rule-based methods are limited to only the predicted rules. Thus, we propose a hybrid strategy to use rules and gnn in unison. We propose a GNN framework based on R-GCN which is trained over the rule path subgraphs.This hybrid strategy i.e., GNN + Rule boosts the performance pf link prediction in inductive setting by a considerable margin.  
## Dependencies Installation ##
You can install the required dependencies using either conda or pip. IndKGC shall work with Python >= 3.7 and PyTorch >= 1.8.0.

## Using Conda ##
```bash
conda install pytorch=1.8.0 cudatoolkit=11.1 pyg -c pytorch -c pyg -c conda-forge
```

## Using Pip ##
```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
```


## Dataset ##
IndKGC is evaluated over three inductive benchmark datasets namely, FB15k-237, WN18RR, and NELL-995 provided by the seminal paper [Inductive Relation Prediction by Subgraph Reasoning](https://github.com/kkteru/grail). All of these datasets have been divided into four versions v1, v2, v3, and v4. The training and validation graphs are extracted from data_version (e.g., fb237_v1) and the test datasets such as fact graph (train_ind) and test_ind are extracted from data_version_ind (e.g., fb237_v1_ind) similar to [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG).

## Pre-requisite Information ##
## 1. AnyBURL ##
IndKGC requires [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) rules and predictions over train, valid, and test_ind for each dataset used.  

## 2. NBFNet Rank & Test Score ##
IndKGC require [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG) ranks and model scores for test data. The NBFNet code should be modified to store the ranks and test data scores in the same order of the test data considered. For example, IndKGC orders the test triplet first and then the corresponding inverse triplets (first <h,r,t> followed by <t, inv_r, h>) from train_ind. Thus, to maintain the order, NBFNet code should be modified to store the ranking and scores in the similar fashion.

## Reproduction ##
The main results in the IndKGC are obtained using the following steps:  
a.0. Get the rule path instantiations for CompGCN/RGCN for the triplets of train, valid, and test_ind datasets.
```bash
   python3 script/get_rule_paths.py -d data/fb15k237_v1 -r anyburl-22/fb15k237_v1 
```
a.1. Get noisy-or confience for Test data
```bash
   python3 script/get_noisy_or.py -d data/fb15k237_v1 -r anyburl-22/fb15k237_v1 
```
b. Train and Test CompGCN/RGCN/Noisy-OR model over the rule paths.
```bash
   python3 script/run.py -d data/fb15k237_v1
```
```bash
   python3 script/run_rgcn.py -d data/fb15k237_v1
```
```bash
   python3 script/run_noisy.py -d data/fb15k237_v1
```

c. The above execution will evaluate Noisy-OR and train and evaluate CompGCN/R-GCN, CompGCN/R-GCN + NBFNet, and NBFNet + NBFNet models. To get evaluation for AnyBURL and AnyBURL + NBFNet execute the below command
```bash
   python3 script/eval_anyburl.py --data_dir data/fb15k237_v1 -r anyburl-22/fb15k237_v1
```
d. The NBFNet evaluation can be obtained by running [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG).


If you make use of this code in your work, please cite the following paper:

	@article{IndKGC,
	  title={Inductive Knowledge Graph Completion with GNN and Rule: An Analysis},
	  author={Akash Anil, Victor Gutierrez Basulto, Yazmín Ibáñez-García, Steven Schockaert},
	  journal={Arxiv.},
	  year={2023}
	}
