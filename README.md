# Project Title
Community detection by fine-tuned optimization of modularity
and modularity density

# Description
This repo comprises two community detection algorithms, which perform fine-tuned
optimization of modularity and modularity density, respectively,
of a community network structure. The fine-tuned algorithm iteratively
carries out splitting and merging stages, alternatively, until
neither splitting nor merging of the community structure
improves the desired metric.

Python implementations of fine-tuned optimizations of modularity and modularity density are in 'ModularityDensity/fine_tuned_modularity.py' and
'ModularityDensity/fine_tuned_modularity_density.py', respectively;
'ModularityDensity/metrics.py' comprises implementation of the metrics
modularity and modularity density.

# Requirements
Python >= 3.7.0,
networkx >= 2.2,
numpy >= 1.15.1,
scipy >= 1.1.0,
nose >= 1.3.7

# Notes
The fine-tuned algorithm is found in [1]. This algorithm works for both
weighted and unweighted, undirected graphs only. The mathematical expressions
of modularity and modularity density are given by equations 35 and 39,
respectively, in [1].

# References
[1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization of
modularity and its variants. IEEE Transactions on Computational Social Systems.
1(1), 46–65, 2014
