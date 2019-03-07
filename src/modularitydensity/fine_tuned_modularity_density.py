# -*- coding: utf-8 -*-
# Copyright (c) CKM Analytix Corp. All rights reserved.
# Authors: Swathi M. Mula (smula@ckmanalytix.com)

"""
Functions for detecting communities based on modularity density
"""

import networkx as nx
from networkx.utils import not_implemented_for
import numpy as np
from networkx.linalg.algebraicconnectivity import fiedler_vector

try:
    from modularitydensity.metrics import modularity_density
except:
    from metrics import modularity_density


__all__ = ['fine_tuned_clustering_qds']


def split_communities_qds(adj, c, normalize, evd_method, tolerence, seed):
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

    # Array of unique community labels
    unique_clusters = np.unique(c)

    # Tracks the nodes in each community
    dict_bool = {}

    # Tracks the clusters that are connected to each community
    dict_connected = {}

    for label in unique_clusters:
        # Track the nodes in each community
        dict_bool[label] = (c == label)

        # Initialize each key to an empty set
        dict_connected[label] = set()

    # Track the clusters that are connected to each community
    for comm1 in unique_clusters[:-1]:
        # index of the community 'comm1'
        i = np.where(unique_clusters == comm1)[0][0]
        bool_1 = dict_bool[comm1]
        adj_comm1 = adj[bool_1]

        # Track the clusters that are connected to community 'comm1'
        for comm2 in unique_clusters[i+1:]:
            bool_2 = dict_bool[comm2]
            zero = np.zeros(len(c), dtype=int)
            zero[bool_2] = 1

            # Check if 'comm2' is connected to 'comm1'
            if ((adj_comm1.dot(zero)).sum()) != 0:
                dict_connected[comm1].add(comm2)
                dict_connected[comm2].add(comm1)

    # Determine the contribution of each community to modularity density
    comm_metric = np.array([modularity_density(adj, c,
                            [cluster_num], dict_bool,
                            np.array(list(dict_connected[cluster_num])))
                            for cluster_num in unique_clusters])

    # Create a copy of cluster labels
    c_new = c.copy()

    # Split each community further if it improves modularity density
    for cluster_num in unique_clusters:

        bool_r = dict_bool[cluster_num]

        # Sparse adjacency matrix corresponding to 'cluster_num'
        sub_adj = adj[bool_r].T[bool_r]

        # Subgraph constructed from sparse adjacency matrix of 'cluster_num'
        g = nx.from_scipy_sparse_matrix(sub_adj)
        # Number of nodes in 'g'
        len_g = len(g)

        # Don't consider further splitting singleton communities
        # or a community which has disconnected modules
        if ((len_g == 1) | (not(nx.is_connected(g)))):
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

            # Modularity density metric value for 'cluster_num'
            curr_metric = comm_metric[unique_clusters == cluster_num][0]

            # Records the splitting information
            split_info = {}

            # Create a copy of the latest cluster labels
            c_latest = c_new.copy()

            # Create a copy of 'dict_bool'
            dict_bool_copy = dict_bool.copy()

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

                    # Array of the union of connected clusters of the
                    # split communities of 'cluster_num'
                    conn_clusters = \
                        np.array(list(((dict_connected[cluster_num]) |
                                 set([cluster_num, new_label]))))

                    # Update the cluster labels in 'c_latest'
                    c_latest[bool_r] = c_sub

                    # Update the boolean array of the split communities
                    # of 'cluster_num'
                    dict_bool_copy[cluster_num] = (c_latest == cluster_num)
                    dict_bool_copy[new_label] = (c_latest == new_label)

                    # Calculate the difference in modularity density
                    # for splitting 'cluster_num'
                    div_metric = (modularity_density(adj,
                                  c_latest, np.unique(c_sub[0:]),
                                  dict_bool_copy, conn_clusters) - curr_metric)

                    # Record the split only if it
                    # improves the modularity density
                    if div_metric > 0:
                        split_info[div_metric] = j

            # Delete to save memory
            del c_latest
            del dict_bool_copy

            # Check if atleast one instance of splitting 'cluster_num'
            # exists that improves modularity density
            if len(split_info) > 0:
                # Split 'cluster_num' based on the division that
                # best improves modularity density
                best_split = split_info[max(split_info.keys())]
                c_sub[nodeIds[0:best_split+1]] = cluster_num
                c_sub[nodeIds[best_split+1:]] = max(c_new) + 1

                # Update 'c_new' with new community labels as a
                # result of splitting 'cluster_num'
                c_new[bool_r] = c_sub

    # Array of community labels, as a result of splitting, for the nodes
    # in the graph as ordered by the adjacency matrix
    return c_new


def merge_communities_qds(adj, c):
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

    # Array of unique community labels
    unique_clusters = np.unique(c)

    # Tracks the nodes in each community
    dict_bool = {}

    # Tracks the clusters that are connected to each community
    dict_connected = {}

    for label in unique_clusters:
        # Track the nodes in each community
        dict_bool[label] = (c == label)

        # Initialize each key to an empty set
        dict_connected[label] = set()

    # Records the merging information
    merging_info = {}

    # Tracks communities, which are connected to atleast one other community
    unique_clusters2 = []
    for comm in unique_clusters:
        bool_1 = dict_bool[comm]
        zero = np.zeros(adj.shape[0], dtype=int)
        zero[~bool_1] = 1

        # Check if the community 'comm' is connected to
        # atleast one other community
        if (adj[bool_1].dot(zero)).sum() != 0:
            # Record the community 'comm'
            unique_clusters2.append(comm)

    # Convert the list of community labels to array
    unique_clusters2 = np.array(unique_clusters2)

    # Track the clusters that are connected to each community
    for comm1 in unique_clusters2[:-1]:
        # index of the community 'comm1'
        i = np.where(unique_clusters2 == comm1)[0][0]
        bool_1 = dict_bool[comm1]
        adj_comm1 = adj[bool_1]

        # Track the clusters that are connected to community 'comm1'
        for comm2 in unique_clusters2[i+1:]:
            bool_2 = dict_bool[comm2]
            zero = np.zeros(len(c), dtype=int)
            zero[bool_2] = 1

            # Check if 'comm2' is connected to 'comm1'
            if ((adj_comm1.dot(zero)).sum()) != 0:
                dict_connected[comm1].add(comm2)
                dict_connected[comm2].add(comm1)

    # Determine the contribution of each community to modularity density
    comm_metric = np.array([modularity_density(adj, c,
                           [cluster_num], dict_bool,
                           np.array(list(dict_connected[cluster_num])))
                           for cluster_num in unique_clusters2])

    # Record the improvement in modularity density for
    # each pair of connected clusters
    for comm1 in unique_clusters2[:-1]:
        # Modularity density for community 'comm1'
        metric_1 = comm_metric[unique_clusters2 == comm1][0]

        # index of the community 'comm1'
        i = np.where(unique_clusters2 == comm1)[0][0]
        bool_1 = dict_bool[comm1]

        # Prospective merger communities of 'comm1'
        for comm2 in unique_clusters2[i+1:]:

            # boolean indices of 'comm2'
            bool_2 = dict_bool[comm2]

            # Consider merging only if 'comm2' is connected to 'comm1'
            if (comm2 in dict_connected[comm1]):

                # Create a copy of cluster labels
                c_latest = c.copy()

                # Create a copy of 'dict_bool'
                dict_bool_copy = dict_bool.copy()

                # Modularity density value for community 'comm2'
                metric_2 = comm_metric[unique_clusters2 == comm2][0]

                # Label of the merged community
                merged_label = min(comm1, comm2)

                # Update the array of community labels to determine the
                # new value of the Modularity density (as a result of merging)
                c_latest[bool_1 | bool_2] = merged_label

                # Update the boolean array of the merged community
                dict_bool_copy[merged_label] = (bool_1 | bool_2)

                # Array of connected clusters of the new merged community
                conn_clusters = np.array(list(((dict_connected[comm1] |
                                         dict_connected[comm2]) -
                                         set([comm1, comm2]))))

                # Calculate the difference in modularity density for
                # merging 'comm1' and 'comm2'
                div_metric = modularity_density(adj, c_latest,
                                                [merged_label],
                                                dict_bool_copy,
                                                conn_clusters) - \
                                               (metric_1 + metric_2)

                # Record the above merge only if it improves modularity density
                if div_metric > 0:
                    merging_info[div_metric] = (comm1, comm2)

                # Delete to save memory
                del c_latest
                del dict_bool_copy

    # Tracks communities which have already merged
    comms_list = []
    # Create a copy of cluster labels
    c_new = c.copy()

    # Check if atleast one instance of merging exists that
    # improves modularity density
    if (len(merging_info) > 0):
        # Sort the merging_info in the descending order of 'div_metric'
        for div_metric in sorted(merging_info.keys(), reverse=True):

            # Consider each pair of clusters, which improve
            # modularity density when merged
            (comm1, comm2) = merging_info[div_metric]

            # Check if 'comm1' or 'comm2' already exist in
            # the list of merged clusters
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
def fine_tuned_clustering_qds(G, normalize=True,
                              evd_method='lanczos',
                              tolerence=1e-08, seed=None):
    r"""Find communities in graph using
       fine-tuned, modularity density maximization. This method supports
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
    >>> c = fine_tuned_clustering_qds(G)
    >>> c
    array([4, 4, 4, 4, 2, 2, 2, 4, 3, 3, 2, 4, 4, 4, 3, 3, 2, 4, 3, 4, 3, 4,
       3, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3])

    Notes
    -----
    The fine-tuned algorithm is found in [1]_. This algorithm works for both
    weighted and unweighted, undirected graphs only.

    Modularity density in [1]_ is given as
    .. math::
        Q = \sum_{c_i \in C}\left [ \frac{|E_{c_i}^{in}|}{|E|}d_{c_i} -
            \left (\frac{2|E_{c_i}^{in}| +
            |E_{c_i}{out}|}{2|E|}d_{c_i} \right )^2 -
            \sum_{c_j \in C, c_j \neq c_i}
            \frac{|E_{c_i, c_j}|}{2|E|}d_{c_i,c_j}   \right ],

        d_{c_i} = \frac{2|E_{c_i}^{in}|}{|c_i|\left ( |c_i| - 1 \right )},

        d_{c_i,c_j} = \frac{|E_{c_i, c_j}|}{|c_i||c_j|}.

    where $C$ is the set of all communities. $c_i$ is a specific community in
    $C$, $|E_{c_i}^{in}|$ is the total weight of edges between nodes within
    community $c_i$, $|E_{c_i}{out}|$ is the total weight of edges from
    nodes in community $c_i$ to the nodes outside $c_i$, and $|E|$ is the
    total weight of edges in the network. $d_{c_i}$ is the internal community
    density of community $c_i$, $d_{c_i, c_j}$ is the pair-wise density between
    communities $c_i$ and $c_j$.

    References
    ----------
    .. [1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization
           of modularity and its variants. IEEE Transactions on Computational
           Social Systems. 1(1), 46–65, 2014

    """

    # Initialize the array of community labels
    c_total = np.zeros(len(G), dtype=int)

    # Perform modularity density maximization for
    # each connected component in 'G'
    for gr in nx.connected_component_subgraphs(G):
        # Nodes of the subgraph 'gr'
        nodes_gr = list(gr)

        # Initialize the array of community labels for the nodes in 'gr'
        c_new = np.zeros(len(nodes_gr), dtype=int)

        # Initialize the number of communities in 'gr'
        comm_size = 1

        # Initialize the number of communities formed by
        # splitting the clusters in 'gr'
        split_size = 0

        # Initialize the number of communities formed by
        # merging the clusters in 'gr'
        merge_size = 0

        # Sparse Adjacency matrix of 'gr'
        adj_gr = nx.to_scipy_sparse_matrix(gr, format='csr')

        # Iteratively carrying out splitting and merging, alternatively,
        # until neither splitting nor merging of the community structure of
        # 'gr' improves modularity density. No improvement in
        # modularity density is indicated if the community structure
        # does not change in one full iteration of splitting and merging
        while ((comm_size != split_size) | (comm_size != merge_size)):

            # Number of communities before splitting and merging
            comm_size = len(np.unique(c_new))

            # Integer array of community labels of the
            # nodes in 'gr' after splitting
            c_new = split_communities_qds(adj_gr, c_new, normalize,
                                          evd_method, tolerence, seed)

            # Number of communities after splitting
            split_size = len(np.unique(c_new))

            # Integer array of community labels of the nodes
            # in 'gr' after merging
            c_new = merge_communities_qds(adj_gr, c_new)

            # Number of communities after merging
            merge_size = len(np.unique(c_new))

        # Update the community labels of the nodes corresponding
        # to 'gr' in original graph 'G'
        c_total[nodes_gr] = np.max(c_total) + 1 + c_new

    # Resultant list of community labels for the nodes in the graph 'G' as
    # ordered by G.nodes()
    return c_total
