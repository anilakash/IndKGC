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
IndKGC require NBFNet's ranks and model scores for test data. The NBFNet code should be modified to store the ranks and test data scores in the same order of the test data considered. For example, IndKGC orders first the test triplet and then the corresponding inverse triplets (first <h,r,t> followed by <t, inv_r, h>). Thus, to maintain the order indices, NBFNet code should be modified to store the ranking and scores in the similar fashion.     
All the results of IndKGC can be reproduced by following the following commands:
```bash

```
