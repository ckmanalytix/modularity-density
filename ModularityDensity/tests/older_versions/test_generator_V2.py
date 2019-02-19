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

sys.path.insert(0, '../tests')
import helpers


class TestFootballQds(object):
    def setup(self):
        self.adj = (helpers.load_football()[0]).tocsr()
        self.G = nx.from_scipy_sparse_matrix(self.adj)

        self.c = np.array([2,  8,  3,  4,  2,  4,  3,  9,  9,  2,  4, 11,  7,
                           3,  7,  3,  2, 5,  7,  1,  5,  9,  9,  2, 11,  8,
                           7,  5, 11,  1,  1,  7,  3,  8,  7,  1, 12,  8,  7,
                           3,  4,  2,  7,  7, 10,  8,  6,  3, 10,  6, 11,
                           9,  4,  6,  7,  1,  5, 10, 12, 12,  3,  7,  5, 12,
                           3,  5, 10,  6,  9, 11,  5,  7,  4,  6,  4, 10,  5,
                           9,  9,  1,  1,  4,  1,  6,  4,  7, 10,  5,  6,  8,
                           11, 10, 10,  2,  1,  5,  5, 12,  4,  7,  3,  1,
                           4,  8,  2,  8,  3,  4,  9,  8,  6,  9, 10,  5,  6])

        self.expected = []
        for label in np.unique(self.c):
            self.expected.append(frozenset(np.where(self.c == label)[0]))
        self.expected = set(self.expected)

        def _check_communities(self, communities):
            assert_equal(self.expected, communities)

        def test_fine_tuned_modularity(self):

            community_array = fine_tuned_clustering_qds(self.G,
                                                        normalize=True,
                                                        evd_method='lanczos',
                                                        tolerence=1e-08,
                                                        seed=100)

            communities = []
            for label in np.unique(community_array):
                communities.append(frozenset(np.where(community_array
                                   == label)[0]))

            _check_communities(set(communities))

        def test_modularity_density(self):

            computed_metric = modularity_density(self.adj, self.c,
                                                 np.unique(self.c))

            assert_equal(round(computed_metric, 4), 0.4944)


class TestFootballQ(object):
    def setup(self):
        self.adj = (helpers.load_football()[0]).tocsr()
        self.G = nx.from_scipy_sparse_matrix(self.adj)

        self.c = np.array([6,  9,  8,  4,  6,  6,  8,  2,  2,  6,  4,  6,  7,
                           8,  7,  8,  6, 5,  7,  3,  5,  2,  2,  6,  6,  9,
                           7,  5,  6,  3,  3,  7,  8,  9,  7,  3,  7,  9,  7,
                           8,  4,  6,  7,  7,  1,  9, 10,  8,  1, 10,  6,  2,
                           4, 10,  7,  3,  5,  1, 10,  5,  8,  7,  5,  5,  8,
                           5,  1,  6,  2,  6,  5,  7,  4, 10,  4,  1,  5,  2,
                           2,  3,  3,  4,  8, 10,  4,  7,  1,  5, 10,  9,  6,
                           1,  1,  6,  3,  5,  5, 10,  4,  7,  8,  3,  4,  9,
                           6,  9,  8,  4,  2,  9, 10,  2,  1,  5, 10])

        self.expected = []
        for label in np.unique(self.c):
            self.expected.append(frozenset(np.where(self.c == label)[0]))
        self.expected = set(self.expected)

        def _check_communities(self, communities):
            assert_equal(self.expected, communities)

        def test_fine_tuned_modularity(self):

            community_array = fine_tuned_clustering_q(self.G, r=0,
                                                      normalize=True,
                                                      evd_method='lanczos',
                                                      tolerence=1e-08,
                                                      seed=100)

            communities = []
            for label in np.unique(community_array):
                communities.append(frozenset(np.where(community_array
                                   == label)[0]))

            _check_communities(set(communities))

        def test_modularity(self):

            computed_metric = modularity_r(self.adj, self.c,
                                           np.unique(self.c), r=0)

            assert_equal(round(computed_metric, 4), 0.5809)
