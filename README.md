Modularity Density
==================
[![Build Status](https://travis-ci.org/ckmanalytix/modularity-density.svg?branch=master)](https://travis-ci.org/ckmanalytix/modularity-density)

<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="99" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="#e05d44" d="M63 0h36v20H63z"/>
        <path fill="url(#b)" d="M0 0h99v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="80" y="15" fill="#010101" fill-opacity=".3">34%</text>
        <text x="80" y="14">34%</text>
    </g>
</svg>

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

Python implementations of fine-tuned optimizations of modularity and modularity density are in 'ModularityDensity/fine_tuned_modularity.py' and
'ModularityDensity/fine_tuned_modularity_density.py', respectively;
'ModularityDensity/metrics.py' comprises implementation of the metrics
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
