# IndKGC: Inductive Knowledge Graph Completion With GNN and Rule #

This is the official implemetation of the paper 'paper_name_with_link'.

Authors:
[Akash Anil, Victor Gutierrez Basulto, Yazmín Ibáñez-García, Steven Schockaert]

## Overview ##

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

## Reproduction ##

## Dataset ##
IndKGC is evaluated over three inductive benchmark datasets namely, FB15k-237, WN18RR, and NELL-995 provided by the seminal paper [Inductive Relation Prediction by Subgraph Reasoning](https://github.com/kkteru/grail). All of these datasets have been divided into four versions v1, v2, v3, and v4. The training and validation graphs are extracted from data_version (e.g., fb237_v1) and the test datasets such as fact graph (train_ind) and test_ind are extracted from data_version_ind (e.g., fb237_v1_ind) similar to [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG).

## Pre-requisite Information ##
## AnyBURL ##
IndKGC requires [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) rules and predictions over train, valid, and test_ind for each dataset used.  

## NBFNet Rank & Test Score ##
IndKGC require [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG) ranks and model scores for test data. The NBFNet code should be modified to store the ranks and test data scores in the same order of the test data considered. For example, IndKGC orders the test triplet first and then the corresponding inverse triplets (first <h,r,t> followed by <t, inv_r, h>) from train_ind. Thus, to maintain the order, NBFNet code should be modified to store the ranking and scores in the similar fashion.

The main results in the IndKGC are obtained in two steps:  
a. Get the rule path instantiations for the triplets of train, valid, and test_ind datasets.
```bash
   
   
```
