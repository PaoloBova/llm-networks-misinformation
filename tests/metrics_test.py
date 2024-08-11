import networkx as nx
import pandas as pd
import pytest
import src.metrics as metrics

class TestMetrics:
    @staticmethod
    def set_up():
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        return graph

    def test_compute_avg_path_length(self):
        graph = self.set_up()
        result = metrics.compute_avg_path_length(graph)
        assert result == 1

    def test_compute_clustering_coefficient(self):
        graph = self.set_up()
        result = metrics.compute_clustering_coefficient(graph)
        assert result == 1.0

    def test_compute_degree_distribution(self):
        graph = self.set_up()
        result = metrics.compute_degree_distribution(graph)
        expected = pd.Series([2, 2, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_connected_components(self):
        graph = self.set_up()
        result = metrics.compute_connected_components(graph)
        assert result == 1

    def test_compute_diameter(self):
        graph = self.set_up()
        result = metrics.compute_diameter(graph)
        assert result == 1

class TestAdvancedMetrics:

    def test_compute_cascade_depth(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        assert metrics.compute_cascade_depth(graph, 1) == 1
        with pytest.raises(AssertionError):
            metrics.compute_cascade_depth(graph, 4)

    def test_compute_cascade_breadth(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        subgraph = graph.subgraph([1, 2])
        assert metrics.compute_cascade_breadth(graph, subgraph) == 2/3
        assert metrics.compute_cascade_breadth(nx.Graph(), subgraph) == 0

    def test_compute_structural_virality(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        assert metrics.compute_structural_virality(graph) == 1
        assert metrics.compute_structural_virality(nx.Graph()) == 0

    def test_compute_fractional_resilience(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        assert metrics.compute_fractional_resilience(graph, [1, 2]) == 2/3
        assert metrics.compute_fractional_resilience(nx.Graph(), [1, 2]) == 0
        with pytest.raises(AssertionError):
            metrics.compute_fractional_resilience(graph, [1, 2, 3, 4])

    def test_compute_time_based_resilience(self):
        assert metrics.compute_time_based_resilience([0.6, 0.7, 0.8]) == 1
        assert metrics.compute_time_based_resilience([0.4, 0.5, 0.6]) == 0
        with pytest.raises(AssertionError):
            metrics.compute_time_based_resilience([])

    def test_compute_topological_resilience(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        assert metrics.compute_topological_resilience(graph, [1]) == 1 - 1/3
        assert metrics.compute_topological_resilience(graph, []) == 1
        assert metrics.compute_topological_resilience(nx.Graph(), [1]) == 0