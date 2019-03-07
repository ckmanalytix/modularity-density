Modularity Density
==================
[![Build Status](https://travis-ci.org/ckmanalytix/modularity-density.svg?branch=master)](https://travis-ci.org/ckmanalytix/modularity-density)

Community detection by fine-tuned optimization of modularity
and modularity density

Description
-----------

This repo comprises two community detection algorithms, which perform fine-tuned
optimization of modularity and modularity density, respectively,
of a community network structure. The fine-tuned algorithm iteratively
carries out splitting and merging stages, alternatively, until
neither splitting nor merging of the community structure
improves the desired metric.

Also included are extensions of the fine_tuned optimizations of both
modularity and modularity density. These extended versions account for the
constraint on the maximum community size, while optimizing the desired metric.

Python implementations of the original fine-tuned optimizations of modularity
and modularity density are in 'src/modularitydensity/fine_tuned_modularity.py' and
'src/modularitydensity/fine_tuned_modularity_density.py', respectively.

Where the extended algorithms are concerned, python implementations of the
constrained versions (setting a threshold on maximum community size) of
fine-tuned optimizations of modularity and modularity density are
in 'src/modularitydensity/constrained_fine_tuned_modularity.py' and
'src/modularitydensity/constrained_fine_tuned_modularity_density.py', respectively.

'src/modularitydensity/metrics.py' comprises implementation of the metrics
modularity and modularity density.

Requirements
------------

python >= 3.5.0,
networkx >= 2.2,
numpy >= 1.15.1,
scipy >= 1.1.0,

Notes
-----

The fine-tuned algorithm is found in [1]. This algorithm works for both
weighted and unweighted, undirected graphs only. The mathematical expressions
of modularity and modularity density are given by equations 35 and 39,
respectively, in [1].

References
----------
[1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization of
modularity and its variants. IEEE Transactions on Computational Social Systems.
1(1), 46–65, 2014
