# -*- coding: utf-8 -*-A
# Copyright (c) CKM Analytix Corp. All rights reserved.
# Authors: Gerardo Veltri (gveltri@ckmanalytix.com), Swathi M. Mula (smula@ckmanalytix.com)

"""
Functions for detecting communities based on modularity density
"""

import itertools
import networkx as nx
from networkx.utils import not_implemented_for
import numpy as np
from networkx.linalg.algebraicconnectivity import fiedler_vector

try:
    from modularitydensity.metrics import mula_modularity_density
except:
    from metrics import mula_modularity_density


__all__ = ['fine_tuned_clustering_mqds']


def split_communities_mqds(adj, c, normalize, evd_method, tolerence, seed):
    """Splits the communities in graph if the splitting
       improves modularity density.

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph.
    c : Integer array
        Current array of community labels for the nodes in the graph as
        ordered by the adjacency matrix.
    normalize : bool
        Whether the normalized Laplacian matrix is used.
    evd_method : string
        Method of eigenvalue computation. It should be one of 'tracemin'
        (TraceMIN), 'lanczos' (Lanczos iteration) and 'lobpcg' (LOBPCG).
    tolerence : float
        Tolerance of relative residual in eigenvalue computation.
    seed : integer, random_state, or None
        Indicator of random number generation state.

    Returns
    -------
    Integer array
        Array of community labels, as a result of splitting, for the nodes
        in the graph as ordered by the adjacency matrix.

    """

    unique_clusters = np.unique(c)
    dict_bool = {}
    curr_modularity = mula_modularity_density(adj, c)
    curr_c  = c.copy()
    split_info = []
    split = False

    for label in unique_clusters:
        # Track the nodes in each community
        dict_bool[label] = (c == label)

    for cluster_num in unique_clusters:
        bool_r = dict_bool[cluster_num]
        sub_adj = adj[bool_r].T[bool_r]
        g = nx.from_scipy_sparse_matrix(sub_adj)
        connected = nx.is_connected(g)
        len_g = sub_adj.shape[0]

        if len_g == 1:
            continue
        elif not connected:
            print("Warning: Check your data as an earliar iteration \
                      resulted in a cluster with \
                      internal disconnected components")
            continue

        f_vector = fiedler_vector(g, weight='weight', normalized=normalize,
                                  tol=tolerence, method=evd_method, seed=seed)

        sub_index = np.arange(len_g)
        nodeIds = [i for f_vector, i in sorted(zip(f_vector, sub_index),
                                               reverse=False,
                                               key=lambda x: x[0])]

        first_community = []
        second_community = []
        second_community.extend(nodeIds)
        c_sub = np.zeros(len_g, dtype=int)
        dict_bool_copy = dict_bool.copy()

        for idx in range(len_g-1):
            first_community.append(second_community.pop())
            g1 = g.subgraph(first_community)
            g2 = g.subgraph(second_community)

            if(nx.is_connected(g1) & nx.is_connected(g2)):
                c_sub[first_community] = cluster_num
                new_label = max(curr_c) + 1
                c_sub[second_community] = new_label

                scratch_c = c.copy()
                scratch_c[bool_r] = c_sub

                split_value = mula_modularity_density(adj, scratch_c)

                if split_value > curr_modularity:
                    split_info.append((split_value, scratch_c))

        if len(split_info) > 0:
            split = True
            curr_c = max(split_info, key=lambda x: x[0])[1]

    return split, curr_c
        
def merge_communities_mqds(adj, c):
    """Merges the communities in graph if the merging improves modularity density.

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph.
    c : Integer array
        Current array of community labels for the nodes in the graph as
        ordered by the adjacency matrix.

    Returns
    -------
    Integer array
        Array of community labels, as a result of merging, for the nodes
        in the graph as ordered by the adjacency matrix.

    """

    curr_c = c.copy()
    unique_clusters = np.unique(c)
    dict_bool = {}
    merge_info = []
    curr_modularity = mula_modularity_density(adj, c)

    for label in unique_clusters:
        dict_bool[label] = (c == label)

    # could potentially be expensive if C approaches N
    for label1, label2 in itertools.combinations(unique_clusters, 2):
        bool1 = dict_bool[label1]
        bool2 = dict_bool[label2]
        if adj[bool1].T[bool2].sum() > 0:
            scratch_c = c.copy()
            merged_label = min(label1, label2)
            scratch_c[bool1 | bool2] = merged_label

            merge_value = mula_modularity_density(adj, scratch_c)

            if merge_value > curr_modularity:
                merge_info.append((merge_value, (label1, label2)))

    merged_labels = []

    # iteratively merge, highest modularity pairings first
    for merge_value, labels in sorted(merge_info, key=lambda x: x[0],
                                      reverse=True):
        if (labels[0] not in merged_labels) & (labels[1] not in merged_labels):
            bool1 = dict_bool[labels[0]]
            bool2 = dict_bool[labels[1]]
            merged_label = min(labels)
            curr_c[bool1] = merged_label
            curr_c[bool2] = merged_label

            # exclude from future merges
            merged_labels.append(labels[0])
            merged_labels.append(labels[1])

    merged = len(merge_info) > 0
    return merged, curr_c

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def fine_tuned_clustering_mqds(G, normalize=True,
                              evd_method='lanczos',
                              tolerence=1e-08, seed=None):
    r"""Find communities in graph using
       fine-tuned, modularity density maximization with a less biased
       modularity metric [2]. This method supports
       weighted/unweighted, undirected graphs only
       and does not support directed graphs.

    The fine-tuned algorithm in [1] iteratively carries out
    splitting and merging stages, alternatively, until
    neither splitting nor merging of the community structure
    improves modularity density

    Parameters
    ----------
    G : NetworkX graph
        Weighted/unweighted, undirected graph.
    normalize : bool, optional
        Whether the normalized Laplacian matrix is used. Default value: True.
    evd_method : string, optional
        Method of eigenvalue computation. It should be one of 'tracemin'
        (TraceMIN), 'lanczos' (Lanczos iteration) and 'lobpcg' (LOBPCG).
        Default value: 'lanczos'.
    tolerence : float, optional
        Tolerance of relative residual in eigenvalue computation. Default
        value: 1e-8.
    seed : integer, random_state, or None, optional
        Indicator of random number generation state. Default value: None.

    Returns
    -------
    Integer array
        Array of community labels for the nodes in the graph as
        ordered by G.nodes().

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = fine_tuned_clustering_mqds(G)
    >>> c
    array([2, 2, 2, 2, 3, 3, 3, 2, 1, 1, 3, 2, 2, 2, 1, 1, 3, 2, 1, 2, 1, 2,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    Notes
    -----
    The fine-tuned algorithm is found in [1]_. The metric is found in [2].
    This algorithm works for both
    weighted and unweighted, undirected graphs only.

    References
    ----------
    .. [1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization
           of modularity and its variants. IEEE Transactions on Computational
           Social Systems. 1(1), 46–65, 2014

    .. [2] MULA S, VELTRI G. A new measure of modularity density for 
           community detection. arXiv:1908.08452 2019.

    """

    
    c_total = np.zeros(len(G), dtype=int)

    # Perform modularity density maximization for
    # each connected component in 'G'
    for gr in nx.connected_component_subgraphs(G):
        nodes_gr = list(gr)
        c_new = np.zeros(len(nodes_gr), dtype=int)

        # Sparse Adjacency matrix of 'gr'
        adj_gr = nx.to_scipy_sparse_matrix(gr, format='csr')

        # Iteratively carrying out splitting and merging, alternatively,
        # until neither splitting nor merging of the community structure of
        # 'gr' improves modularity density.
        split = True
        merged = True
        while split | merged:
            split, c_new = split_communities_mqds(adj_gr, c_new, normalize,
                                                  evd_method, tolerence,
                                                  seed)
            merged, c_new = merge_communities_mqds(adj_gr, c_new)


        # Update the community labels of the nodes corresponding
        # to 'gr' in original graph 'G'
        c_total[nodes_gr] = np.max(c_total) + 1 + c_new

    return c_total
