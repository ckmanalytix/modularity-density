"""Unit tests for the fine_tuned_clustering_qds.py, fine_tuned_clustering_q.py,
   modularity_density.py, modularity_r.py modules
"""
import networkx as nx
import numpy as np
from nose.tools import assert_equal

import sys
sys.path.append("..")

from metrics import modularity_density
from metrics import modularity_r
from fine_tuned_modularity_density \
 import fine_tuned_clustering_qds
from fine_tuned_modularity \
 import fine_tuned_clustering_q


class TestQds(object):

    def setup(self):
        self.G = nx.karate_club_graph()
        self.adj = nx.to_scipy_sparse_matrix(self.G)

        gr_john = np.array([8, 9, 14, 15, 18, 20, 22,
                           23, 26, 27, 28, 29, 30, 32, 33])

        gr_hi = np.array([0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21])

        gr_3 = np.array([24, 25, 31])

        gr_4 = np.array([4, 5, 6, 10, 16])

        self.c = np.zeros(self.adj.shape[0], dtype=int)
        self.c[gr_john] = 0
        self.c[gr_hi] = 1
        self.c[gr_3] = 2
        self.c[gr_4] = 3

        self.expected = set({frozenset(gr_john), frozenset(gr_hi),
                            frozenset(gr_3), frozenset(gr_4)})

    def _check_communities(self, communities):
        assert_equal(self.expected, communities)

    def test_fine_tuned_modularity_density(self):

        community_array = fine_tuned_clustering_qds(self.G, normalize=True,
                                                    evd_method='lanczos',
                                                    tolerence=1e-08, seed=100)

        communities = []
        for label in np.unique(community_array):
            communities.append(frozenset(np.where(community_array
                                                  == label)[0]))

        self._check_communities(set(communities))

    def test_modularity_density(self):

        computed_metric = modularity_density(self.adj, self.c,
                                             np.unique(self.c))

        assert_equal(round(computed_metric, 4), 0.2313)


class TestQ(object):

    def setup(self):
        self.G = nx.karate_club_graph()
        self.adj = nx.to_scipy_sparse_matrix(self.G)

        gr_john = np.array([8, 9, 14, 15, 18, 20, 22,
                            26, 29, 30, 32, 33])

        gr_hi = np.array([0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21])

        gr_3 = np.array([23, 24, 25, 27, 28, 31])

        gr_4 = np.array([4, 5, 6, 10, 16])

        self.c = np.zeros(self.adj.shape[0], dtype=int)
        self.c[gr_john] = 0
        self.c[gr_hi] = 1
        self.c[gr_3] = 2
        self.c[gr_4] = 3

        self.expected = set({frozenset(gr_john), frozenset(gr_hi),
                            frozenset(gr_3), frozenset(gr_4)})

    def _check_communities(self, communities):
        assert_equal(self.expected, communities)

    def test_fine_tuned_modularity(self):

        community_array = fine_tuned_clustering_q(self.G, r=0,
                                                  normalize=True,
                                                  evd_method='lanczos',
                                                  tolerence=1e-08, seed=100)

        communities = []
        for label in np.unique(community_array):
            communities.append(frozenset(np.where(community_array
                                                  == label)[0]))

        self._check_communities(set(communities))

    def test_modularity(self):

        computed_metric = modularity_r(self.adj, self.c,
                                       np.unique(self.c), r=0)

        assert_equal(round(computed_metric, 4), 0.4198)


class TestQRescaled(object):

    def setup(self):
        self.G = nx.karate_club_graph()
        self.adj = nx.to_scipy_sparse_matrix(self.G)

        gr_john = np.array([8, 14, 15, 18, 20, 22, 26, 29, 30, 32, 33])

        gr_hi = np.array([0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21])

        gr_3 = np.array([23, 24, 25, 27, 28, 31])

        gr_4 = np.array([4, 5, 6, 10, 16])

        gr_5 = np.array([9])

        self.c = np.zeros(self.adj.shape[0], dtype=int)
        self.c[gr_john] = 0
        self.c[gr_hi] = 1
        self.c[gr_3] = 2
        self.c[gr_4] = 3
        self.c[gr_5] = 4

        self.expected = set({frozenset(gr_john), frozenset(gr_hi),
                            frozenset(gr_3), frozenset(gr_4), frozenset(gr_5)})

    def _check_communities(self, communities):
        assert_equal(self.expected, communities)

    def test_fine_tuned_modularity_r(self):

        community_array = fine_tuned_clustering_q(self.G, r=2.0,
                                                  normalize=True,
                                                  evd_method='lanczos',
                                                  tolerence=1e-08, seed=100)

        communities = []
        for label in np.unique(community_array):
            communities.append(frozenset(np.where(community_array
                               == label)[0]))

        self._check_communities(set(communities))

    def test_modularity_r(self):

        computed_metric = modularity_r(self.adj, self.c,
                                       np.unique(self.c), r=2.0)

        assert_equal(round(computed_metric, 4), 0.5148)
