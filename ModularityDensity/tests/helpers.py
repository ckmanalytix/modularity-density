import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx
import gc
import pickle
import pandas as pd
import math
#from networkx.algorithms.community import greedy_modularity_communities



def _save(file_name, obj):
    with open(file_name, 'wb') as fobj:
        pickle.dump(obj, fobj)

def _load(file_name):
    with open(file_name, 'rb') as fobj:
        return pickle.load(fobj)

def get_sample(adj_id, comm_id):
    return _load(adj_id), _load(comm_id)

### Make adj mat ###
def generate_adj_matrix(path, kwargs, first_node_id, undirected=True):
    """Read data and make adjacency matrix.

    Parameters
    ----------
    path: str
        path to .txt file
    kwargs: dict
        specification for how to load data
    first_node_id: int
        id of first node (e.g. either 0 or 1)
    undirected: bool
        if True, (aRb ^ bRa not in R) then bRa inserted into R

    Returns
    -------
    adj_matrix: numpy array
        n x n adjacency matrix
    """

    n1, n2 = np.loadtxt(path, **kwargs)

    if first_node_id != 0:
        n1 = n1 - first_node_id
        n2 = n2 - first_node_id

    n = max(max(n1), max(n2)) + 1

    #Initialize matrix
    adj_mat = sparse.lil_matrix((n,n))

    #Fill in matrix
    zipped = np.array([n1, n2]).T
    for node1, node2 in zipped:
        adj_mat[node1, node2] = 1
        if undirected: adj_mat[node2, node1] = 1

    return adj_mat

### Load datasets ####
def load_karate_club():
    params = {
        'comments': '%',
        'converters': {0: lambda x: int(x),
                       1: lambda x: int(x)},
        'unpack': True,
        'dtype': int
    }

    path = '../data/zach/out.ucidata-zachary'
    adj = generate_adj_matrix(path, params, first_node_id=1, undirected=True)

    print('shape of adj matrix:', adj.shape)

    #from http://www1.ind.ku.dk/complexLearning/zachary1977.pdf
    mr_hi = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22]
    john = [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

    nodes = []
    for node in range(adj.shape[0]):
        comm = 1 if node + 1 in mr_hi else 0
        nodes.append({
            'gtc_community': comm,
            'gtc_color': 'red' if comm == 1 else 0,
        })

    color_map = ['C0', 'C1', 'C2', 'C3', 'C4']

    #nx greedy q
    #greedy_q = greedy_modularity_communities(nx.Graph(adj)) #this output a list of sets

    #for node in range(adj.shape[0]):
    #    for i in range(len(greedy_q)):
    #        if node in greedy_q[i]:
    #            nodes.append({'greedyQ_community': i,
    #            'greedyQ_color': color_map[i % len(color_map)],
    #    })
    #        else: continue

    #fine-tuned q
    #community hard-coded in for now to mininize stuffs needed to install to run this helpers
    #but maybe ideally this would call swathi's code
    ft_q = [3, 3, 3, 3, 1, 1, 1, 3, 2, 2, 1, 3, 3, 3, 2, 2, 1, 3, 2, 3, 2, 3, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 2, 2]

    for node in range(adj.shape[0]):
        nodes.append({
            'ftQ_community': ft_q[node],
            'ftQ_color': color_map[node % len(color_map)],
        })


    #fine-tuned qds
    ft_qds = [3, 3, 3, 3, 1, 1, 1, 3, 2, 2, 1, 3, 3, 3, 2, 2, 1, 3, 2, 3, 2, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2]

    for node in range(adj.shape[0]):
        nodes.append({
            'ftQds_community': ft_qds[node],
            'ftQds_color': color_map[node % len(color_map)],
        })

    return adj, nodes

def viz_karate_club(adj, color_map=None):
    G = nx.Graph(adj)
    label_keys = {}

    for i in list(np.arange(adj.shape[0])):
        label_keys[i] = i+1

    plt.figure(1, figsize=(12,10))
    pos = nx.spring_layout(G, k=(.1/(math.sqrt(adj.shape[0]))), seed=100)
    nx.draw_networkx_edges(G, pos, color='k',width=1, alpha=0.5)
    if color_map is not None:
        nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=.5, width=0.5, node_size=100)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='k', alpha=.5, width=0.5, node_size=100)
    nx.draw_networkx_labels(G, pos, labels=label_keys, font_size=21, alpha=5)
    plt.title('Zacharys Karate Club', fontsize=20)

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)

    plt.show()
    return None

def load_DBLP(sample=False, sample_size=None):
    if sample:
        if sample_size is None:
            raise('error: specify sample size')
        else:
            adj_sample, sample_mapping_df = make_DBLP_samples(thres=sample_size, visualize=False, image_path=None)

            return adj_sample, sample_mapping_df

    else:
        params = {'comments': '#',
                  'delimiter': '\t',
                  'unpack': True,
                  'dtype': int}

        #load adj mat
        path = '../data/dblp/com-dblp.ungraph.txt'
        adj = generate_adj_matrix(path, params, first_node_id=0, undirected=True)
        print('DBLP adj matrix shape:', adj.shape)

        #load comm
        path_comm = '../data/dblp/com-dblp.all.cmty.txt'
        all_comm = list()

        with open(path_comm, 'rb') as f:
            for line in f:

                line = str(line)

                for char in ['\\n', 'b', '\'']:
                    line = line.replace(char,'')

                newlist = line.split('\\t')
                all_comm.append(newlist)

        print('number of comm:', len(all_comm))

        return adj, all_comm

def make_DBLP_5k(save=False):
    """Make a connected sample of size 5k"""
    params = {'comments': '#',
         'delimiter': '\t',
         'unpack': True,
         'dtype': int}

    #data
    address = '../data/com-dblp.ungraph.txt'
    adj = generate_adj_matrix(address, params, first_node_id=0)

    #communities
    address_comm = '../data/com-dblp.all.cmty.txt'

    all_comm = list()

    counter = 0
    with open(address_comm, 'rb') as f:
        for line in f:
            if counter == 0:
                print(line)
                counter = counter + 1

            line = str(line)

            for char in ['\\n', 'b', '\'']:
                line = line.replace(char,'')

            newlist = line.split('\\t')
            all_comm.append(newlist)

    #get sizes
    sizes = list()

    for comm in all_comm:
        sizes.append(len(comm))

    find_5k = np.asarray([np.abs(x - 5000) for x in sizes])
    print('number of nodes in sample:', sizes[find_5k.argmin()])

    #isolate sample
    sample_nodes = all_comm[find_5k.argmin()]
    #needs to map str to int else will throw an error
    sample_nodes = list(map(int, sample_nodes))
    adj_sample = adj[sample_nodes,:][:, sample_nodes]

    if save:
        helpers._save('../data/DBLP_samples/dblp_' + str(adj_sample.shape[0]) + '_adj', adj_sample)

    return adj_sample

def load_DBLP_5k():
    return _load('../data/DBLP_samples/dblp_4975_adj')

def make_DBLP_samples(thres, visualize=False, image_path=None):
    """Make samples, visualize ground truth, save results"""

    params = {'comments': '#',
         'delimiter': '\t',
         'unpack': True,
         'dtype': int}

    print('load adj matrix...')
    path = '../data/dblp/com-dblp.ungraph.txt'
    adj = generate_adj_matrix(path, params, first_node_id=0)

    print('original adj matrix shape:', adj.shape)

    path_comm = '../data/dblp/com-dblp.all.cmty.txt'
    all_comm = list()

    with open(path_comm, 'rb') as f:
        for line in f:

            line = str(line)

            for char in ['\\n', 'b', '\'']:
                line = line.replace(char,'')

            newlist = line.split('\\t')
            all_comm.append(newlist)

    print('original number of comm:', len(all_comm))

    # pick random comms until number of unique nodes hit thres
    comm_counter = 0
    node_counter = 0
    sample_mapping = {'node': [], 'community': [], 'node_in_sample': [], 'color': []}
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    sample_nodes = list()
    skipped = 0

    while len(sample_nodes) < thres:
        choice = np.random.choice(all_comm, replace=False)
        all_comm.remove(choice) #prevent comm repeats

        #add to dict
        for node_i in choice:
            if node_i not in sample_nodes:
                sample_nodes.append(node_i)
                sample_mapping['node'].append(node_i)
                sample_mapping['node_in_sample'].append(node_counter)
                sample_mapping['community'].append(comm_counter)
                sample_mapping['color'].append(c_list[comm_counter % len(c_list)])

                node_counter = node_counter + 1
            else: skipped = skipped + 1

        comm_counter = comm_counter + 1

    print('number of sample nodes:', len(sample_nodes))
    print('number of clusters:', comm_counter)
    print('number of node repeats (excluded):', skipped)

    #convert to int for indexing
    sample_nodes = list(map(int, sample_nodes))

    print('take adj matrix sample...')
    adj_sample = adj[sample_nodes,:][:, sample_nodes]

    sample_mapping_df = pd.DataFrame(sample_mapping)
    sample_mapping_df = sample_mapping_df.sort_values(by='node_in_sample')

    #save
    print('pickling sample...')
    _save('../data/dblp/dblp_samples/dblp_' + str(adj_sample.shape[0]) + '_adj', adj_sample)
    _save('../data/dblp/dblp_samples/dblp_' + str(adj_sample.shape[0]) + '_comm', sample_mapping_df)

    if visualize:
        visualize_DBLP(adj_sample, sample_mapping_df, image_path)

    return adj_sample, sample_mapping_df


def visualize_DBLP(adj_sample, sample_mapping_df, image_path):
    print('visualizing network...')
    adj_sample_size = adj_sample.shape
    G = nx.from_scipy_sparse_matrix(adj_sample)

    fig1 = plt.figure(1, figsize=(80,60))
    pos = nx.spring_layout(G, scale=0.8, k=(2/(math.sqrt(adj_sample_size[0]))))
    nx.draw_networkx_edges(G, pos, color='k',width=3, alpha=.2)
    nx.draw_networkx_nodes(G, pos, node_color=sample_mapping_df['color'].values.tolist(),
                        alpha=0.4, width=20, node_size=300)

    if image_path is not None:
        fig1.savefig(image_path, bbox_inches='tight')
    else:
        fig1.savefig('../viz/DBLP_sample_' + str(adj_sample_size[0]) + '.png', bbox_inches='tight')
    plt.close()
    return None

def load_football(weighted=True, visualize_ground_truth=False):
    def _generate_adj_matrix_football(path, kwargs, first_node_id, weighted=weighted):

        n1, n2, weight = np.loadtxt(path, **kwargs)

        if first_node_id != 0:
            n1 = [x - first_node_id for x in n1]
            n2 = [x - first_node_id for x in n2]

        n = max(max(n1), max(n2)) + 1

        #Initialize matrix
        adj_mat = sparse.lil_matrix((n,n))

        counter = 0
        weighted_nodes = [(85-1, 4-1), (100-1, 15-1), (28-1, 18-1)]

        #Fill in matrix
        if weighted:
            for node1, node2 in zip(n1, n2):
                if (node1, node2) in weighted_nodes:
                    adj_mat[node1, node2] = 2
                    adj_mat[node2, node1] = 2

                else:
                    adj_mat[node1, node2] = 1
                    adj_mat[node2, node1] = 1

                counter = counter + 1

        else:
            for node1, node2 in zip(n1, n2):
                    adj_mat[node1, node2] = 1
                    adj_mat[node2, node1] = 1
                    counter = counter + 1

        return adj_mat

    #load adjacency matrix
    params = {'comments': '%',
         'delimiter': ' ',
         'unpack': True,
         'dtype': int}

    print('loading matrix...')
    path = '../tests/data/football/football.mtx'
    adj = _generate_adj_matrix_football(path, params, first_node_id=1, weighted=weighted)

    #load ground truth
    path_comm = '../tests/data/football/football_nodevalue.mtx'
    node_comm = list()

    with open(path_comm, 'rb') as f:
        for line in f:
            line = str(line)
            if '%' in line: continue #skip comment rows
            for char in ['\\n', 'b', '\'']:
                line = line.replace(char,'')

            node_comm.append(int(line))

    if visualize_ground_truth:

        #map color
        color_map = list()
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'darkgreen', 'indigo', 'tomato']

        for comm_i in node_comm:
            color_map.append(colors[comm_i % len(colors)])

        #visualize ground truth
        print('visualizing...')
        viz_football(adj, color_map)

    return adj, node_comm

def viz_football(adj, color_map=None):
    #load node name
    path_name = '../data/football/football_nodename.txt'

    node_name = list()

    with open(path_name, 'rb') as f:
        for line in f:
            line = str(line)
            for char in ['\\n', 'b', '\'']:
                line = line.replace(char,'')

            node_name.append(line)

    G = nx.from_scipy_sparse_matrix(adj)

    #assign labels
    label_keys = {}

    for i in list(np.arange(adj.shape[0])):
        label_keys[i] = node_name[i]

    #render
    fig1 = plt.figure(1, figsize=(50,40))
    pos = nx.spring_layout(G, scale=0.8, k=(1/(math.sqrt(adj.shape[0]))), seed=100)

    nx.draw_networkx_edges(G, pos, color='k',width=3, alpha=.2)
    if color_map is not None:
        nx.draw_networkx_nodes(G, pos, node_color=color_map,
                            alpha=.75, width=100, node_size=2000)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='g',
                            alpha=.75, width=100, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=label_keys, font_size=30, alpha=3)
    plt.show()
    return None

def load_pgp():
    params = {'comments': '%',
         'delimiter': ',',
         'unpack': True,
         'dtype': int}

    path = '../data/pgp/arenas-pgp.edges'
    adj = generate_adj_matrix(path, params, first_node_id=1, undirected=True)

    return adj

def load_asl():
    G = nx.read_gml('../data/aslevel/as-22july06.gml')
    return nx.adjacency_matrix(G)

###Community-based layout###
#Code from: (cite stackoverflow article)
#Doesn't work as well as adjusting weight edges + spring layout, so skip the rest of the code
def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def test_community():
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    from community import community_louvain

    g = nx.karate_club_graph()
    partition = community_louvain.best_partition(g)
    print(partition)
    pos = community_layout(g, partition)

    nx.draw(g, pos, node_color=list(partition.values())); plt.show()
    return
