"""Unit tests for the fine_tuned_clustering_qds.py, fine_tuned_clustering_q.py,
   modularity_density.py, modularity_r.py modules
"""
import networkx as nx
import numpy as np
from nose.tools import assert_equal
from nose.tools import raises

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

    def test_fine_tuned_modularity_density(self):

        community_array = fine_tuned_clustering_qds(self.G, normalize=False,
                                                    seed=100)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2382)

    def test_fine_tuned_modularity_density_normalized(self):

        community_array = fine_tuned_clustering_qds(self.G, seed=100)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2313)

    def test_fine_tuned_modularity_density_method2(self):

        community_array = fine_tuned_clustering_qds(self.G,
                                                    evd_method='tracemin',
                                                    seed=100)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2313)

    def test_fine_tuned_modularity_density_method3(self):

        community_array = fine_tuned_clustering_qds(self.G,
                                                    evd_method='lobpcg',
                                                    seed=100)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2313)

    def test_fine_tuned_modularity_density_tolerence(self):

        community_array = fine_tuned_clustering_qds(self.G, tolerence=1e-04,
                                                    seed=100)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2313)

    def test_fine_tuned_modularity_density_seed(self):

        community_array = fine_tuned_clustering_qds(self.G, seed=2)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2313)

    def test_fine_tuned_modularity_density_2(self):

        community_array = fine_tuned_clustering_qds(self.G, normalize=False,
                                                    evd_method='lobpcg',
                                                    seed=2)

        computed_metric = modularity_density(self.adj, community_array,
                                             np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.2382)

    def test_fine_tuned_modularity(self):

        community_array = fine_tuned_clustering_q(self.G, normalize=False,
                                                  seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_normalized(self):

        community_array = fine_tuned_clustering_q(self.G, seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_method2(self):

        community_array = fine_tuned_clustering_q(self.G,
                                                  evd_method='tracemin',
                                                  seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_method3(self):

        community_array = fine_tuned_clustering_q(self.G,
                                                  evd_method='lobpcg',
                                                  seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_tolerence(self):

        community_array = fine_tuned_clustering_q(self.G, tolerence=1e-04,
                                                  seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_r(self):

        community_array = fine_tuned_clustering_q(self.G, r=2.0, seed=100)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array), r=2.0)

        assert_equal(round(computed_metric, 4), 0.5148)

    def test_fine_tuned_modularity_seed(self):

        community_array = fine_tuned_clustering_q(self.G, seed=2)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array))

        assert_equal(round(computed_metric, 4), 0.4198)

    def test_fine_tuned_modularity_2(self):

        community_array = fine_tuned_clustering_q(self.G, normalize=False,
                                                  r=2, evd_method='lobpcg',
                                                  seed=2)

        computed_metric = modularity_r(self.adj, community_array,
                                       np.unique(community_array), r=2)

        assert_equal(round(computed_metric, 4), 0.5153)

    @raises(nx.NetworkXNotImplemented)
    def test_fine_tuned_modularity_density_exception1(self):
        Gr = nx.DiGraph()
        fine_tuned_clustering_qds(Gr)

    @raises(nx.NetworkXNotImplemented)
    def test_fine_tuned_modularity_density_exception2(self):
        Gr = nx.MultiGraph()
        fine_tuned_clustering_qds(Gr)

    @raises(nx.NetworkXNotImplemented)
    def test_fine_tuned_modularity_exception1(self):
        Gr = nx.DiGraph()
        fine_tuned_clustering_qds(Gr)

    @raises(nx.NetworkXNotImplemented)
    def test_fine_tuned_modularity_exception2(self):
        Gr = nx.MultiGraph()
        fine_tuned_clustering_qds(Gr)
