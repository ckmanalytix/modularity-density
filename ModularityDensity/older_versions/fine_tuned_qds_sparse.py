"""
Functions for detecting communities based on modularity density
"""

import networkx as nx
import numpy as np
from networkx.linalg.algebraicconnectivity import fiedler_vector
from metrics_sparse import *
import timeit

def split_communities_qds(adj, c, normalize, evd_method, tolerence):
    """Short summary.

    Parameters
    ----------
    adj : type
        Description of parameter `adj`.
    c : type
        Description of parameter `c`.
    normalize : type
        Description of parameter `normalize`.
    evd_method : type
        Description of parameter `evd_method`.
    tolerence : type
        Description of parameter `tolerence`.

    Returns
    -------
    type
        Description of returned object.

    """
    """This algorithm will split each of the cluster into two further subclusters if the split improves the modularity density metric
        param adj: Similarity matrix (Sparse matrix format: csr or csc)
        param c: integer array of community labels
        param normalise: Would you like to perform normalized cut based on Laplacian Matrix?
        param tolerence: tolerence limit in the determination of Eigen values/vectors of the Laplacian Matrix

        returns c_new: resultant integer array of community labels as ordered by the adjacency matrix
    """
    #Array of unique cluster labels
    unique_clusters = np.unique(c)

    print("Splitting {}".format(len(np.unique(c))))

    #Creating a dictionary, with cluster labels as keys and corresponding boolean array (c == label) as values
    dict_bool = {}
    #Creating a dictionary, with cluster labels as keys and corresponding labels of the connected components as values
    dict_connected = {}

    for label in unique_clusters:
        dict_bool[label] = (c == label)
        dict_connected[label] = set()

    #Creating dictionary values for the dict_connected
    for comm1 in unique_clusters[:-1]:
         #index of the current community
         i = np.where(unique_clusters == comm1)[0][0]
         bool_1 = dict_bool[comm1]#(c == comm1)
         adj_comm1 = adj[bool_1]

         #Prospective merger communities of comm1
         for comm2 in unique_clusters[i+1:]:

             #boolean indices of comm2 labels
             bool_2 = dict_bool[comm2]#(c == comm2)

             #Considering only those combinations which result in connected graph with comm1
             #Bascially, we don't consider disconnected communities
             zero = np.zeros(len(c), dtype = int)
             zero[bool_2] = 1

             if ((adj_comm1.dot(zero)).sum()) != 0:
                 dict_connected[comm1].add(comm2)
                 dict_connected[comm2].add(comm1)

    #Getting the metric value for each community in the orginal graph
    comm_metric = np.array([modularity_density(adj, c, np.array([cluster_num]), dict_bool, np.array(list(dict_connected[cluster_num]))) for cluster_num in unique_clusters])

    #Creating a copy of the cluster labels
    c_new = c.copy()


    #Splitting each community further if it improves the metric
    for cluster_num in unique_clusters:

        bool_r = dict_bool[cluster_num] #(c == cluster_num)

        #Adjacency matrix corresponding to cluster_num
        sub_adj = adj[bool_r].T[bool_r]

        #SubGraph constructed from sparse Adjacency matrix of cluster_num
        g = nx.from_scipy_sparse_matrix(sub_adj)

        #Don't consider further splitting singleton communities or a disconnected community
        if (((sub_adj).sum() == 0) | (not(nx.is_connected(g)))):
            if(not(nx.is_connected(g))):
                print("Warning: check your metric or data as an earliar splitting resulted the cluster {} with internal disconnected subgraphs".format(cluster_num))
            continue
        else:
            #print("The cluster num is {}".format(cluster_num))

            #Creating a new community label array for the root cluster_num community
            c_sub = np.zeros(sub_adj.shape[0], dtype = int)

            #new indices for the above matrix
            sub_index = np.arange(len(c_sub))

            #Determining the fiedler_vector
            f_vector = fiedler_vector(g, weight='weight', normalized = normalize, tol= tolerence, method= evd_method)

            #Reaaranging the elements based on the order of fieldler vector
            nodeIds = [i for f_vector,i in sorted(zip(f_vector, sub_index), reverse = True)]

            first_community = []
            second_community = []
            second_community.extend(nodeIds)

            #Metric value for the current community in the original graph
            curr_metric = comm_metric[unique_clusters == cluster_num][0]

            #Form |c| divisions and record the best one
            split_info = {}

            #Creating a copy of the latest labels
            c_latest = c_new.copy()

            #Creating a copy of the dict_bool
            dict_bool_copy = dict_bool.copy()

            #Possible splits according to the fielder vector
            for j in range(len(nodeIds)-1):

                #Splitting the cluster_num community into two clusters
                first_community.append(nodeIds[j])
                second_community.remove(nodeIds[j])

                g1 = g.subgraph(first_community)
                g2 = g.subgraph(second_community)

                if(nx.is_connected(g1) & nx.is_connected(g2)):
                    #Relabelling the community cluster_num
                    c_sub[first_community] = cluster_num
                    new_label = max(c_new) + 1
                    c_sub[second_community] = new_label


                    #Array of the Resulting union of connected communities of the two splitted sub-clusters
                    conn_clusters = np.array(list(((dict_connected[cluster_num]) | set([cluster_num, new_label]))))
                    #print(conn_clusters)

                    #Updating the labels in c_latest
                    #c_latest[c == cluster_num] = c_sub
                    c_latest[bool_r] = c_sub

                    #Updating the boolean array of the split keys in the copy of dict_bool
                    dict_bool_copy[cluster_num] = (c_latest == cluster_num)
                    dict_bool_copy[new_label] = (c_latest == new_label)

                    #Calculating the difference in the metric for the above split
                    div_metric = (modularity_density(adj, c_latest, np.unique(c_sub[0:]), dict_bool_copy, conn_clusters) - curr_metric)
                    #print("The div metric, curr_metric are {} and {}".format(div_metric, curr_metric))

                    #Record the split only if it improves the metric value
                    if div_metric > 0:
                        split_info[div_metric] = j

            #Deleting to save memory
            del c_latest
            del dict_bool_copy

            #Checking if we have atleast one split (in cluster_num) that improves the metric
            if len(split_info) > 0:
                #Finding the best split, which best improves the metric
                best_split = split_info[max(split_info.keys())]
                c_sub[nodeIds[0:best_split+1]] = cluster_num
                c_sub[nodeIds[best_split+1:]] = max(c_new) + 1

                #Updating the cluster labels of the above nodes in c_new
                #c_new[c == cluster_num] = c_sub
                c_new[bool_r] = c_sub

    #print("The resultant splitting is {}".format(c_new))

    #resulting integer array of community labels as ordered by the adjacency matrix
    return c_new

def merge_communities_qds(adj, c):
    """Merges the communities in graph if the merging improves modularity density.

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph of interest.
    c : Integer array
        Current array of community labels for the nodes in the graph as ordered by the adjacency matrix.

    Returns
    -------
    Integer array
        Array of community labels, as a result of merging, for the nodes in the graph as ordered by the adjacency matrix.

    """
s
    #Array of unique community labels
    unique_clusters = np.unique(c)

    #Tracks the nodes in each community
    dict_bool = {}

    #Tracks the clusters that are connected to each community
    dict_connected = {}

    for label in unique_clusters:
        #Track the nodes in each community
        dict_bool[label] = (c == label)

        #Initialize each key to an empty set
        dict_connected[label] = set()

    #Records the merging information
    merging_info = {}

    #Tracks communities, which are not disconnected to other communities
    unique_clusters2 = []
    for comm in unique_clusters:
        bool_1 = dict_bool[comm]
        zero = np.zeros(adj.shape[0], dtype = int)
        zero[~bool_1] = 1

        #Check if the community is not a disconnected component
        if (adj[bool_1].dot(zero)).sum() != 0:
            #Record the community
            unique_clusters2.append(comm)

    #Convert the list of community labels to array
    unique_clusters2 = np.array(unique_clusters2)

    #Track the clusters that are connected to each community
    for comm1 in unique_clusters2[:-1]:
         #index of the current community
         i = np.where(unique_clusters2 == comm1)[0][0]
         bool_1 = dict_bool[comm1]
         adj_comm1 = adj[bool_1]

         #Track the clusters that are connected to community comm1
         for comm2 in unique_clusters2[i+1:]:
             bool_2 = dict_bool[comm2]
             zero = np.zeros(len(c), dtype = int)
             zero[bool_2] = 1

             #Check if comm2 is connected to comm1
             if ((adj_comm1.dot(zero)).sum()) != 0:
                 dict_connected[comm1].add(comm2)
                 dict_connected[comm2].add(comm1)

    #Determine the contribution of each community (in the current community structure) to modularity density
    comm_metric = np.array([modularity_density(adj, c, np.array([cluster_num]), dict_bool, np.array(list(dict_connected[cluster_num]))) for cluster_num in unique_clusters2])

    #Record the improvement in modularity density for each pair of connected clusters
    for comm1 in unique_clusters2[:-1]:
        #Metric value for community comm1 in the orginal community structure
        metric_1 = comm_metric[unique_clusters2 == comm1][0]

        #index of the current community
        i = np.where(unique_clusters2 == comm1)[0][0]
        bool_1 = dict_bool[comm1]

        #Prospective merger communities of comm1
        for comm2 in unique_clusters2[i+1:]:

            #boolean indices of comm2
            bool_2 = dict_bool[comm2]

            #Consider merging only if comm2 is connected to comm1
            if (comm2 in dict_connected[comm1]):

                #Create a copy of cluster labels
                c_latest = c.copy()

                #Create a copy of dict_bool
                dict_bool_copy = dict_bool.copy()

                #Metric value for community comm2 in the orginal community structure
                metric_2 = comm_metric[unique_clusters2 == comm2][0]

                #Label of the merged community
                merged_label = min(comm1, comm2)

                #Update the array of community labels to determine the new value of the metric (as a result of merging)
                c_latest[bool_1 | bool_2] = merged_label

                #Update the boolean array of the merged community
                dict_bool_copy[merged_label] = (bool_1 | bool_2)

                #Array of connected clusters of the new merged community
                conn_clusters = np.array(list(((dict_connected[comm1] | dict_connected[comm2]) - set([comm1, comm2]))))

                #Calculate the difference in modularity density for the merge of comm1 and comm2
                div_metric = modularity_density(adj, c_latest, np.array([merged_label]), dict_bool_copy, conn_clusters) - (metric_1 + metric_2)

                #Record the above merge only if it improves modularity density
                if div_metric > 0:
                    merging_info[div_metric] = (comm1, comm2)

                #Delete to save memory
                del c_latest
                del dict_bool_copy

    #Tracks communities which have already merged
    comms_list = []
    #Create a copy of cluster labels
    c_new = c.copy()

    #Checking if atleast one instance of merging exists that improves modularity density
    if (len(merging_info) > 0):
        #Sort the merging_info in the descending order of div_metric
        for div_metric in sorted(merging_info.keys(), reverse = True):

            #Consider each pair of clusters, which improve modularity density when merged
            (comm1, comm2) = merging_info[div_metric]

            #Check if comm1 or comm2 already exist in the list of merged clusters
            if ((not(comm1 in comms_list)) & (not(comm2 in comms_list))):
                #Merge the pair of communities
                comms_list.extend([comm1, comm2])

                #Label of the merged community
                c_new[dict_bool[comm1] | dict_bool[comm2]] = min(comm1, comm2)

    #Array of community labels, as a result of merging, for the nodes in the graph as ordered by the adjacency matrix
    return c_new


def fine_tuned_clustering_qds(adj, normalize = True, evd_method ='lanczos', tolerence = 1e-08):
    """Find communities in graph using fine-tuned, modularity density maximization. This method supports
    weighted/unweighted, undirected graphs only and does not support directed graphs.

    The fine-tuned algorithm iteratively carries out splitting and merging stages, alternatively, until neither splitting nor merging
    of the community structure improves modularity density

    Parameters
    ----------
    adj : SciPy sparse matrix (csr or csc)
        The N x N Adjacency matrix of the graph of interest.
    normalize : bool, optional
        Whether the normalized Laplacian matrix is used. Default value: True.
    evd_method : string, optional
        Method of eigenvalue computation. It should be one of 'tracemin'
        (TraceMIN), 'lanczos' (Lanczos iteration) and 'lobpcg' (LOBPCG).
        Default value: 'lanczos'.
    tolerence : float, optional
        Tolerance of relative residual in eigenvalue computation. Default
        value: 1e-8.

    Returns
    -------
    Integer array
        Array of community labels for the nodes in the graph as ordered by the adjacency matrix.

    References
    ----------
    [1] CHEN M, KUZMIN K, SZYMANSKI BK. Community detection via maximization of modularity
        and its variants. IEEE Transactions on Computational Social Systems. 1(1), 46–65, 2014

    """

    #Graph constructed from the adjacency matrix
    G = nx.from_scipy_sparse_matrix(adj)

    #Initialize the array of community labels
    c_total = np.zeros(adj.shape[0], dtype = int)

    #Perform modularity density maximization for each connected component in G
    for gr in nx.connected_components(G):
        #Nodes of the subgraph
        nodes_gr = list(gr)

        #Initialize the array of community labels for the nodes in the subgraph
        c_new = np.zeros(len(nodes_gr), dtype = int)

        #Initial the number of communities in the subgraph
        comm_size = 1

        #Initialize the number of communities formed by splitting the clusters of the subgraph
        split_size = 0

        #Initialize the number of communities formed by merging the clusters of the subgraph
        merge_size = 0

        #Adjacency matrix of the subgraph
        adj_gr = adj[nodes_gr].T[nodes_gr]

        #Repeatedly carrying out splitting and merging stages, alternatively, until neither splitting nor merging
        #of the community structure of subgraph improves modularity density. No improvement in modularity density
        #is implied if the community structure does not change in one full iteration of splitting and merging
        while ((comm_size != split_size) | (comm_size != merge_size)):

            #Number of communities before splitting and merging
            comm_size = len(np.unique(c_new))

            #Integer array of community labels after splitting
            c_new = split_communities_qds(adj_gr, c_new, normalize, evd_method, tolerence)

            #Number of communities after splitting
            split_size = len(np.unique(c_new))

            #Integer array of community labels after merging
            c_new = merge_communities_qds(adj_gr, c_new)

            #Number of communities after merging
            merge_size = len(np.unique(c_new))

        #Updating community labels of the nodes in original graph G
        c_total[nodes_gr] = np.max(c_total) + 1 + c_new

    #Resultant list of community labels for the nodes in the graph as ordered by the adjacency matrix
    return c_total
