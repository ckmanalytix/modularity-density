# -*- coding: utf-8 -*-
# Copyright (c) CKM Analytix Corp. All rights reserved.
# Authors: Swathi M. Mula (smula@ckmanalytix.com)

"""
Functions for detecting communities based on modularity, while
constrained to a threshold on the maximum size of community
"""

import networkx as nx
from networkx.utils import not_implemented_for
import numpy as np
from networkx.linalg.algebraicconnectivity import fiedler_vector
try:
    from modularitydensity.metrics import modularity_r
    from modularitydensity.fine_tuned_modularity import fine_tuned_clustering_q
except:
    from metrics import modularity_r
    from fine_tuned_modularity import fine_tuned_clustering_q

__all__ = ['constrained_fine_tuned_clustering_q']


def forced_split_communities_q(adj, c, cluster_size, r,
                               normalize, evd_method, tolerence, seed):
    """Force splits the communities in graph, if the size of the first_community
       is greater than the threshold, such that the splitting least
       compromizes modularity.

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph.
    c : Integer array
        Current array of community labels for the nodes in the graph as
        ordered by the adjacency matrix.
    cluster_size : integer
        Threshold/maximum size (number of nodes) of a cluster.
    r : float
        Resolution of the topology: smaller 'r' favors forming larger
        communities, while larger 'r' favors forming smaller communities.
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

    # Array of unique community labels
    unique_clusters = np.unique(c)

    # Tracks the nodes in each community
    dict_bool = {}

    for label in unique_clusters:
        # Track the nodes in each community
        dict_bool[label] = (c == label)

    # Create a copy of cluster labels
    c_new = c.copy()

    # Split each community, whose size is greater than the threshold
    for cluster_num in unique_clusters:

        bool_r = dict_bool[cluster_num]

        # Sparse adjacency matrix corresponding to 'cluster_num'
        sub_adj = adj[bool_r].T[bool_r]

        # Subgraph constructed from sparse adjacency matrix of 'cluster_num'
        g = nx.from_scipy_sparse_matrix(sub_adj)
        # Number of nodes in 'g'
        len_g = len(g)

        # Don't consider further splitting singleton communities
        # or communities of size lower than the threshold
        # or a community which has disconnected modules
        if ((len_g == 1) | (len_g <= cluster_size) |
           (not(nx.is_connected(g)))):

            if(not(nx.is_connected(g))):
                print("Warning: Check your data as an earliar iteration \
                      resulted in a cluster with \
                      internal disconnected components")
            continue
        else:

            # Create an array of community labels for the
            # nodes in 'cluster_num'
            c_sub = np.zeros(len_g, dtype=int)

            # indices of the nodes in 'sub_adj'
            sub_index = np.arange(len_g)

            # Determine the fiedler_vector of subgraph 'g'
            f_vector = fiedler_vector(g, weight='weight', normalized=normalize,
                                      tol=tolerence,
                                      method=evd_method, seed=seed)

            # Rearrange the nodes of 'sub_adj' in the descreasing order of
            # elements of fieldler vector
            nodeIds = [i for f_vector, i in sorted(zip(f_vector, sub_index),
                                                   reverse=True)]

            # Initialize the communities corresponding to
            # bipartitioning of 'cluster_num'
            first_community = []
            second_community = []
            second_community.extend(nodeIds)

            # Records the splitting information
            split_info = {}

            # Create a copy of the latest cluster labels
            c_latest = c_new.copy()

            # Possible splits of 'cluster_num' based on the fielder vector
            for j in range(len(nodeIds)-1):

                # Split the 'cluster_num' into two clusters
                first_community.append(nodeIds[j])
                second_community.remove(nodeIds[j])

                # Graph induced by nodes in 'first_community'
                g1 = g.subgraph(first_community)

                # Graph induced by nodes in 'second_community'
                g2 = g.subgraph(second_community)

                # Check if 'g1' and 'g2' are connected graphs each
                if(nx.is_connected(g1) & nx.is_connected(g2)):
                    # Relabel the cluster labels of nodes in 'cluster_num'
                    c_sub[first_community] = cluster_num
                    new_label = max(c_new) + 1
                    c_sub[second_community] = new_label

                    # Update the cluster labels in 'c_latest'
                    c_latest[bool_r] = c_sub

                    # Tracks the nodes in each of the split communities
                    # of 'cluster_num'
                    dict_bool_copy = dict()
                    dict_bool_copy[cluster_num] = (c_latest == cluster_num)
                    dict_bool_copy[new_label] = (c_latest == new_label)

                    # Calculate the modularity after splitting 'cluster_num'
                    div_metric = modularity_r(adj, c_latest,
                                              np.unique(c_sub[0:]), r,
                                              dict_bool_copy)

                    # Record the split
                    split_info[div_metric] = j

                    # Delete to save memory
                    del dict_bool_copy

            # Delete to save memory
            del c_latest

            # Check if atleast one instance of splitting 'cluster_num' exists
            # that does not result in disconnected modules
            if len(split_info) > 0:
                # Split 'cluster_num' based on the division that
                # least compromizes modularity
                best_split = split_info[max(split_info.keys())]
                c_sub[nodeIds[0:best_split+1]] = cluster_num
                c_sub[nodeIds[best_split+1:]] = max(c_new) + 1

                # Update 'c_new' with new community labels as a
                # result of splitting 'cluster_num'
                c_new[bool_r] = c_sub
            else:
                print("No split possible for cluster num: {}, \
                 as any further split results in disconnected modules".
                      format(cluster_num))

    # Array of community labels, as a result of splitting, for the nodes
    # in the graph as ordered by the adjacency matrix
    return c_new


def constrained_merge_communities_q(adj, c, cluster_size, r):
    """Merges the communities in graph if the merging improves modularity,
       under the condition that the merging does not result in a community
       size greater than the threshold.

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph.
    c : Integer array
        Current array of community labels for the nodes in the graph as
        ordered by the adjacency matrix.
    cluster_size : integer
        Threshold/maximum size (number of nodes) of a cluster.
    r : float
        Resolution of the topology: smaller 'r' favors forming larger
        communities, while larger 'r' favors forming smaller communities.

    Returns
    -------
    Integer array
        Array of community labels, as a result of merging, for the nodes
        in the graph as ordered by the adjacency matrix.

    """

    # Array of unique community labels
    unique_clusters = np.unique(c)

    # Tracks the nodes in each community
    dict_bool = {}

    for label in unique_clusters:
        # Track the nodes in each community
        dict_bool[label] = (c == label)

    # Records the merging information
    merging_info = {}

    # Tracks communities, which are connected to atleast one other community
    unique_clusters2 = []
    for comm in unique_clusters:
        bool_1 = dict_bool[comm]

        zero = np.zeros(adj.shape[0], dtype=int)
        zero[~bool_1] = 1

        # Check if the community 'comm' is connected to
        # atleast one other community and the community size is less than
        # the threshold
        if ((adj[bool_1].dot(zero)).sum() != 0) & \
           (np.count_nonzero(bool_1) < cluster_size):
            # Record the community 'comm'
            unique_clusters2.append(comm)

    # Convert the list of community labels to array
    unique_clusters2 = np.array(unique_clusters2)

    # Determine the contribution of each community to modularity
    comm_metric = np.array([modularity_r(adj, c, [cluster_num],
                           r, dict_bool) for cluster_num in unique_clusters2])

    # Record the improvement in modularity for
    # each pair of connected clusters
    for comm1 in unique_clusters2[:-1]:
        # Modularity value for community 'comm1'
        metric_1 = comm_metric[unique_clusters2 == comm1][0]

        # index of the community 'comm1'
        i = np.where(unique_clusters2 == comm1)[0][0]
        bool_1 = dict_bool[comm1]

        adj_comm1 = adj[bool_1]

        # Prospective merger communities of 'comm1'
        for comm2 in unique_clusters2[i+1:]:
            bool_2 = dict_bool[comm2]
            zero = np.zeros(adj.shape[0], dtype=int)
            zero[bool_2] = 1

            # Consider merging only if 'comm2' is connected to 'comm1', and
            # the merging results in a cluster size less than
            # or equal to the threshold
            if ((adj_comm1.dot(zero)).sum() != 0) & \
               (np.count_nonzero(bool_1 | bool_2) <= cluster_size):

                # Create a copy of cluster labels
                c_latest = c.copy()

                # Modularity value for community 'comm2'
                metric_2 = comm_metric[unique_clusters2 == comm2][0]

                # Label of the merged community
                merged_label = min(comm1, comm2)

                # Update the array of community labels to determine the
                # new value of the Modularity (as a result of merging)
                c_latest[bool_1 | bool_2] = merged_label

                # Create a copy of 'dict_bool'
                dict_bool_copy = dict()

                # Update the boolean array of the merged community
                dict_bool_copy[merged_label] = (bool_1 | bool_2)

                # Calculate the difference in modularity for merging
                # 'comm1' and 'comm2'
                div_metric = modularity_r(adj,
                                          c_latest,
                                          [merged_label], r,
                                          dict_bool_copy) - \
                                         (metric_1 + metric_2)

                # Record the above merge only if it improves modularity
                if div_metric > 0:
                    merging_info[div_metric] = (comm1, comm2)

                # Deleting to save memory
                del dict_bool_copy
                del c_latest

    # Tracks communities which have already merged
    comms_list = []
    # Create a copy of cluster labels
    c_new = c.copy()

    # Check if atleast one instance of merging exists that improves modularity
    if (len(merging_info) > 0):
        # Sort the merging_info in the descending order of 'div_metric'
        for div_metric in sorted(merging_info.keys(), reverse=True):

            # Consider each pair of clusters, which
            # improve modularity when merged
            (comm1, comm2) = merging_info[div_metric]

            # Check if 'comm1' or 'comm2' already exist in the
            # list of merged clusters
            if ((not(comm1 in comms_list)) & (not(comm2 in comms_list))):
                # Merge the pair of communities
                comms_list.extend([comm1, comm2])

                # Label of the merged community
                c_new[dict_bool[comm1] | dict_bool[comm2]] = min(comm1, comm2)

    # Array of community labels, as a result of merging,
    # for the nodes in the graph as ordered by the adjacency matrix
    return c_new


@not_implemented_for('directed')
@not_implemented_for('multigraph')
def constrained_fine_tuned_clustering_q(G, cluster_size, r=0, normalize=True,
                                        evd_method='lanczos',
                                        tolerence=1e-08, seed=None):
    r"""Find communities in graph using modularity maximization
        subjected to the constraint that the community size is less than
        the desired threshold. This method supports weighted/unweighted,
        undirected graphs only and does not support directed graphs.

        This algorithm is carried out in two stages:
            1. In the first stage, the community structure is obtained
               from 'fine_tuned_clustering_q', which finds communities using
               fine-tuned, modularity maximization described in [1].

            2. In the second stage, using the communities obtained from
               stage 1, the condition on the maximum community size is imposed,
               while maximizing modularity.

    Parameters
    ----------
    G : NetworkX graph
        Weighted/unweighted, undirected graph.
    cluster_size : integer
        Threshold/maximum size (number of nodes) of a cluster.
    r : float
        Resolution of the topology: smaller 'r' favors forming larger
        communities, while larger 'r' favors forming smaller communities.
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
    >>> c = constrained_fine_tuned_clustering_q(G, cluster_size=15)
    >>> c
    array([5, 5, 5, 5, 3, 3, 3, 5, 2, 2, 3, 5, 5, 5, 2, 2, 3, 5, 2, 5, 2, 5,
       2, 4, 4, 4, 2, 4, 4, 2, 2, 4, 2, 2])
    >>> c = constrained_fine_tuned_clustering_q(G, cluster_size=10, seed=100)
    >>> c
    array([5, 5, 5, 5, 3, 3, 3, 5, 4, 4, 3, 5, 5, 5, 4, 4, 3, 5, 4, 5, 4, 7,
       4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4])

    Notes
    -----
    Modularity in [1]_,[2]_ is given as
    .. math::

        Q = \sum_{c_i \in C}\left [ \frac{|E_{c_i}^{in}|}{|E|} -
            \left (\frac{2|E_{c_i}^{in}| +
            |E_{c_i}^{out}|}{2|E|}  \right )^2 \right ],

    where $C$ is the set of all communities. $c_i$ is a specific community in
    $C$, $|E_{c_i}^{in}|$ is the total weight of edges between nodes within
    community $c_i$, $|E_{c_i}{out}|$ is the total weight of edges from
    nodes in community $c_i$ to the nodes outside $c_i$, and $|E|$ is the
    total weight of edges in the network.

    Modularity for rescaled topology (see [1]_) at scale $r$ is given as
    .. math::

        Q_r = \sum_{c_i \in C}\left [ \frac{2|E_{c_i}^{in}| +r|c_i|}{2|E| +
              r|V|} - \left (\frac{2|E_{c_i}^{in}| + |E_{c_i}^{out}| +
              r|c_i|}{2|E| + r|V|}  \right )^2 \right ],

    where $|c_i|$ is the number of nodes in a specific community. $|V|$ is the
    total number of nodes in the entire network structure.

    References
    ----------
    .. [1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization
           of modularity and its variants. IEEE Transactions on Computational
           Social Systems. 1(1), 46–65, 2014

    .. [2] NEWMAN MEJ, GIRVAN M. Finding and evaluating community structure in
           community structure in networks. Phys. Rev. E. 69, 026113, 2004

    """

    # Initialize the cluster labels to zero
    c_total = np.zeros(len(G), dtype=int)

    # Perform modularity maximization for
    # each connected component in 'G', such that the community size is
    # less than the threshold
    for gr in nx.connected_component_subgraphs(G):
        # Nodes of the subgraph 'gr'
        nodes_gr = list(gr)

        # Initialize the array of community labels for the nodes in 'gr' to the
        # community structure from 'fine_tuned_clustering_q'
        c_new = fine_tuned_clustering_q(gr, r, normalize=normalize,
                                        evd_method=evd_method,
                                        tolerence=tolerence, seed=seed)

        # Initialize the number of communities in 'gr'
        comm_size = len(np.unique(c_new))

        # Initialize the number of communities formed by
        # splitting the clusters in 'gr'
        split_size = 0

        # Initialize the number of communities formed by
        # merging the clusters in 'gr'
        merge_size = 0

        # Sparse Adjacency matrix of 'gr'
        adj_gr = nx.to_scipy_sparse_matrix(gr, format='csr')

        # Iteratively carrying out splitting and merging, alternatively, until
        # there is no further improvement in modularity, while the
        # community size is less than the threshold. No improvement in
        # modularity under the constraint on the community size
        # is indicated if the community structure does not change in
        # one full iteration of splitting and merging
        while ((comm_size != split_size) | (comm_size != merge_size)):

            # Number of communities before splitting and merging
            comm_size = len(np.unique(c_new))

            # Integer array of community labels of the
            # nodes in 'gr' after splitting
            c_new = forced_split_communities_q(adj_gr, c_new,
                                               cluster_size, r, normalize,
                                               evd_method, tolerence, seed)

            # Number of communities after splitting
            split_size = len(np.unique(c_new))

            # Integer array of community labels of the nodes
            # in 'gr' after merging
            c_new = constrained_merge_communities_q(adj_gr, c_new,
                                                    cluster_size, r)

            # Number of communities after merging
            merge_size = len(np.unique(c_new))

        # Update the community labels of the nodes corresponding
        # to 'gr' in original graph 'G'
        c_total[nodes_gr] = np.max(c_total) + 1 + c_new

    # Resultant list of community labels for the nodes in the graph 'G' as
    # ordered by G.nodes()
    return c_total
