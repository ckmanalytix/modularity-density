[![Build Status](https://travis-ci.org/ckmanalytix/modularity-density.svg?branch=master)](https://travis-ci.org/ckmanalytix/modularity-density) ![Python badge](https://img.shields.io/badge/python-3.5|3.6|3.7-<blue>.svg)

Community detection by fine-tuned optimization of modularity
and modularity density

Dependencies
------------
<table>
<tr>
  <td>Python</td>
  <td>
    <a> >= 3.5.0 
    </a>
  </td>
</tr>
  <td>NetworkX</td>
  <td>
    <a> >= 2.2s
    </a>
</td>
</tr>
<tr>
  <td>NumPy</td>
  <td>
    <a> >= 1.15.1
    </a>
  </td>
</tr>
<tr>
  <td>SciPy</td>
  <td>
    <a> >= 1.1.0
    </a>
  </td>
</tr>
</table>

Installation
-----
```sh
pip install modularitydensity
```

Quick Start
-----
```python
import networkx as nx
import numpy as np
from modularitydensity.metrics import modularity_density
from modularitydensity.fine_tuned_modularity_density import fine_tuned_clustering_qds

G = nx.karate_club_graph() #sample dataset
adj = nx.to_scipy_sparse_matrix(G) #convert to sparse matrix

community_array = fine_tuned_clustering_qds(G)
print(community_array)
>> [2 2 2 2 4 4 4 2 3 3 4 2 2 2 3 3 4 2 3 2 3 2 3 3 1 1 3 3 3 3 3 1 3 3]

computed_metric = modularity_density(adj, community_array, np.unique(community_array))
print(computed_metric)
>> 0.2312650016945721    
```

Description
-----------

This package comprises two community detection algorithms which perform fine-tuned
optimization of modularity and modularity density, respectively,
of a community network structure. The fine-tuned algorithm iteratively
carries out splitting and merging stages, alternatively, until
neither splitting nor merging of the community structure
improves the desired metric.

Also included are extensions of the fine_tuned optimizations of both
modules. These extended versions account for any
constraint on the maximum community size, while optimizing the desired metric.

Source code can be found at: https://github.com/ckmanalytix/modularity-density/

References
----------
[1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization of
modularity and its variants. IEEE Transactions on Computational Social Systems.
1(1), 46–65, 2014
