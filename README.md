# RWK: Regularized Wasserstein Kernel for Graphs

This is a Python3 implementation of RWK. This framework provides a novel optimal transport distance metric, namely Regularized Wasserstein Kernels (RWK) for graphs, that enables fast computations of distances between graphs under the preservation of feature and structural properties.

### Prerequisites

* Torch (>= 1.7.1)
* Networkx (>= 2.2)
* POT [Python Optimal Transport] (>=0.7.0)
* Sklearn (>= 0.19.1)
* Numpy (>=1.19.0)
* Scipy (>=1.2.1)
* Cython (>=0.27.3)
* Matplotlib (>=2.2.2)

### Evaluation and dataset references

We use 12 benchmark datasets grouped in two categories. 
(1) Graphs with discrete attributes
(2) Graphs with continuous attributes
You can find more details of our datasets from `DATASET_DESCRIPTIONS.md` file.

We have evaluated our method using twelve datasets against 16 state-of-the-art baselines. 

The experimental results show that our method consistently outperforms all state-of-the-art methods on all benchmark 
datasets for two different graph settings: 
* graphs with discrete attributes 
* graphs with continuous attributes

We consider three settings in our experiments: 

* `RWK-0` - without using local variations
* `RWK-1` - with using 1-hop local variations
* `RWK` - with using 2-hop local variations (default setting)

We use the nested cross validation setup to evaluate our method.

### Files description

* rjw_cross_validation.ipynb - RWK cross validation (ipython notebook version).
* sinkhorn_algorithms.py - Sinkhorn-knopp algorithm for regularized optimal transport (OT) problem and return the OT matrix.
* scg_optimizer.py - Solve the regularized OT problem with the Sinkhorn Conditional Gradient (SCG) and Line-search algorithms.
* ot_distances.py - Compute Wasserstein distance between source graph and target graph by concatenating the original graph signals with local variations.
* custom_svc.py - Create a SVM classifier over the RW distance using the proposed kernel e^{-\gamma*RW}.
* RJW.py - Compute Gromov-Wasserstein distance and local barycentric Wasserstein distance between two graphs and their gradients.
* graph.py - Compute the structural similarity matrix between two graphs by node embeddings.
* attentionwalk.py - Attention Walk Layer for node embeddings.
* attention_utils - Calculate the probability transition tensor by heat kernel random walk.
* utils.py - Generate log files and other utility functions. 
* data_loader.py - Data preprocessing and loading the data.

### Citation

Please cite our paper if you use this code in your research work.

```
@inproceedings{asiri2021rwk,
  title={A Regularized Wasserstein Framework for Graph Kernels}, 
  author={Asiri Wijesinghe, Qing Wang and Stephen Gould}, 
  booktitle={IEEE International Conference on Data Mining (ICDM)},
  year={2021}
}
```

### License

MIT License

### Contact for RWK issues

Please contact me: asiri.wijesinghe@anu.edu.au if you have any questions / submit a Github issue if you find any bugs.