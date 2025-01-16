# Use this file to add functions for computing metrics useful for analysis


# Let's delve deeper into quantitative measures for information cascade depth and breadth, as well as network resilience, drawing from relevant literature in network science.

# 1. Information Cascade Depth and Breadth

# Information cascades in networks have been studied extensively, particularly in the context of social influence and information diffusion. We can measure cascade depth and breadth as follows:

# a) Cascade Depth:
# Cascade depth represents how far the misinformation spreads through the network. We can measure this using the concept of "generations" introduced by Leskovec et al. (2007) in their study of viral marketing:

# - Depth = max(d_i)

# Where d_i is the shortest path length from the source of misinformation to node i that has adopted the misinformation.

# b) Cascade Breadth:
# Breadth represents how wide the misinformation spreads. We can measure this using the concept of "size" from Goel et al. (2015) in their study of structural virality:

# - Breadth = N_m / N

# Where N_m is the number of nodes that have adopted the misinformation, and N is the total number of nodes in the network.

# c) Structural Virality:
# To capture both depth and breadth in a single measure, we can use the "structural virality" metric proposed by Goel et al. (2015):

# - Structural Virality = (1 / N_m * (N_m - 1)) * Σ_i Σ_j d_ij

# Where d_ij is the shortest path length between nodes i and j in the induced subgraph of adopters.

# This measure captures both the depth and breadth of the cascade, with higher values indicating more viral spread.

# 2. Resilience

# Resilience in the context of maintaining correct information can be measured in several ways:

# a) Fractional Resilience:
# Based on the concept of "network integrity" from Albert et al. (2000), we can define fractional resilience as:

# - R_f = N_c / N

# Where N_c is the number of nodes maintaining correct information after the introduction of misinformation, and N is the total number of nodes.

# b) Time-based Resilience:
# Inspired by the work on temporal networks by Holme and Saramäki (2012), we can measure how long the network maintains a certain level of correct information:

# - R_t = t_θ / t_max

# Where t_θ is the time it takes for the fraction of correct nodes to fall below a threshold θ (e.g., 0.5), and t_max is the total simulation time.

# c) Topological Resilience:
# Drawing from the work on network robustness by Schneider et al. (2011), we can measure how the network's structure contributes to its resilience:

# - R_top = 1 - (S_m / S_0)

# Where S_m is the size of the largest connected component of misinformed nodes, and S_0 is the size of the largest connected component in the original network.

# d) Recovery Rate:
# Based on the concept of "healing" in network epidemiology (Pastor-Satorras et al., 2015), we can measure how quickly the network recovers from misinformation:

# - R_r = (dN_c/dt) / N_m

# Where dN_c/dt is the rate of change in the number of correct nodes, and N_m is the current number of misinformed nodes.


import numpy as np
import pandas as pandas
import networkx as nx
import random
import src.utils

def compute_average_degree(graph):
    degrees = [d for n, d in graph.degree()]
    average_degree = np.mean(degrees)
    return average_degree

def compute_avg_path_length(graph):
    """Compute the average path length of a graph.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    avg_path_length: the average path length of the graph
    """
    avg_path_length = nx.average_shortest_path_length(graph)
    return avg_path_length

def compute_clustering_coefficient(graph):
    """Compute the clustering coefficient of a graph.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    clustering_coefficient: the clustering coefficient of the graph
    """
    clustering_coefficient = nx.average_clustering(graph)
    return clustering_coefficient

def compute_degree_distribution(graph):
    """Compute the degree distribution of a graph.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    degree_distribution: a pandas series of the degree distribution of the graph
    """
    degree_distribution = pandas.Series([val for (node, val) in graph.degree()])
    return degree_distribution

def compute_connected_components(graph):
    """Compute the number of connected components in a graph.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    num_connected_components: the number of connected components in the graph
    """
    num_connected_components = nx.number_connected_components(graph)
    return num_connected_components

def compute_diameter(graph):
    """Compute the diameter of a graph.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    diameter: the diameter of the graph
    """
    diameter = nx.diameter(graph)
    return diameter

def compute_cascade_depth(subgraph, source_node):
    """Compute the depth of an information cascade in a network.
    
    The cascade depth is a measure of how far information has spread from the
    source node in a network. It is computed as the maximum shortest path length
    from the source node to any other node in the subgraph.
    
    Parameters:
    subgraph: a networkx graph object
    source_node: the node ID representing the source of the cascade
    
    Returns:
    depth: the depth of the information cascade
    
    Notes:
    - This function assumes that you pass in the subgraph which contains all
    nodes that have adopted the information.
    - If the source node is not in the subgraph, then this function returns 0.
    This is a limitation of this metric.    
    """
    if source_node not in subgraph:
        return 0
    try:
        depth = max(nx.single_source_shortest_path_length(subgraph, source_node).values())
    except nx.exception.NetworkXNoPath:
        # If the source node or any other node is disconnected from the subgraph
        depth = 0
    return depth

def compute_cascade_breadth(graph, subgraph):
    """Compute the breadth of an information cascade in a network.
    
    The cascade breadth is a measure of how widely information has spread in a
    network. It is computed as the ratio of the number of nodes in the subgraph
    to the number of nodes in the entire graph.
    
    Parameters:
    graph: a networkx graph object
    subgraph: a networkx subgraph object
    
    Returns:
    breadth: the breadth of the information cascade
    """
    if len(graph) == 0:
        return 0
    breadth = len(subgraph) / len(graph)
    return breadth

def compute_structural_virality(subgraph):
    """Compute the structural virality of an information cascade in a network.
    
    The structural virality is a measure of the complexity of an information
    cascade in a network. It is computed as the average shortest path length
    between all pairs of nodes in the subgraph.
    
    The concept of structural virality was introduced in the influential paper
    "The Structural Virality of Online Diffusion" by Goel, Anderson, Hofman,
    and Watts in 2015.

    In their paper, structural virality is defined as the average pairwise
    distance between all nodes in a diffusion tree. In other words, it measures
    the average number of "hops" it takes to get from one randomly chosen
    individual to another in the network of people who have shared a piece of
    information. This is a measure of the complexity of the diffusion process: a
    low structural virality indicates a broadcast pattern where information
    spreads from a central source to many individuals, while a high structural
    virality indicates a viral pattern where information spreads through a long
    chain of person-to-person transmission.
    
    Note that here we allow arbitrary subgraphs to be passed in, not just trees.

    The reason why structural virality only relies on the subgraph and not on
    any information from the full graph is that it is a measure of the structure
    of the diffusion process itself, not of the underlying network.
    
    Parameters:
    subgraph: a networkx subgraph object
    
    Returns:
    structural_virality: the structural virality of the information cascade
    """
    if len(subgraph) < 2:
        return 0
    structural_virality = 0
    for node1 in subgraph.nodes():
        for node2 in subgraph.nodes():
            if node1 != node2:
                # Compute the shortest path length between node1 and node2
                # If there is no path between the nodes, we set the shortest
                # path length to 0. Since we are interested in how far the
                # information has spread, it doesn't make sense to consider the
                # distance between disconnected nodes.
                try:
                    shortest_path_length = nx.shortest_path_length(subgraph, node1, node2)
                except nx.exception.NetworkXNoPath:
                    shortest_path_length = 0
                structural_virality += shortest_path_length
    structural_virality /= len(subgraph) * (len(subgraph) - 1)
    return structural_virality

def compute_fractional_resilience(graph, correct_nodes):
    """Compute the fractional resilience of a network.
    
    The fractional resilience is a measure of a network's resistance to
    misinformation. It is computed as the ratio of the number of nodes with
    correct information to the total number of nodes in the graph.
    
    Parameters:
    graph: a networkx graph object
    correct_nodes: a list of node IDs representing nodes with correct information
    
    Returns:
    fractional_resilience: the fractional resilience of the network
    """
    if len(graph) == 0:
        return 0
    assert len(correct_nodes) <= len(graph)
    fractional_resilience = len(correct_nodes) / len(graph)
    return fractional_resilience

def compute_time_based_resilience(correct_proportions, threshold=0.5):
    """Compute the time-based resilience of a network.
    
    The time-based resilience is a measure of a network's ability to maintain
    correct information over time. It is computed as the ratio of the number of
    time steps where the proportion of correct information is above a certain
    threshold to the total number of time steps.
    
    Parameters:
    correct_proportions: a list of correct proportions at different time steps
    threshold: the threshold proportion for resilience
    
    Returns:
    time_based_resilience: the time-based resilience of the network
    """
    assert len(correct_proportions) > 0
    indices = np.argwhere(np.array(correct_proportions) > threshold).flatten()
    time_based_resilience = len(np.argwhere(indices == np.arange(indices.size))) / len(correct_proportions)
    return time_based_resilience

def compute_topological_resilience(graph, misinformed_nodes):
    """Compute the topological resilience of a network.
    
    The topological resilience is a measure of a network's structural resistance
    to misinformation. It is computed as one minus the ratio of the size of the
    largest connected component of misinformed nodes to the size of the largest
    connected component in the entire graph.
    
    Parameters:
    graph: a networkx graph object
    misinformed_nodes: a list of node IDs representing nodes with misinformation
    
    Returns:
    topological_resilience: the topological resilience of the network
    """
    if len(graph) == 0:
        return 0
    if len(misinformed_nodes) == 0:
        return 1
    misinformed_subgraph = graph.subgraph(misinformed_nodes)
    if len(misinformed_subgraph) == 0:
        return 1
    largest_connected_component = max(nx.connected_components(graph), key=len)
    largest_misinformed_component = max(nx.connected_components(misinformed_subgraph), key=len)
    topological_resilience = 1 - (len(largest_misinformed_component) / len(largest_connected_component))
    return topological_resilience

def compute_recovery_rate(correct_nodes, misinformed_nodes, time_step):
    """Compute the recovery rate of a network.
    
    Parameters:
    correct_nodes: a list of node IDs representing nodes with correct information
    misinformed_nodes: a list of node IDs representing nodes with misinformation
    time_step: the current time step
    
    Returns:
    recovery_rate: the recovery rate of the network
    """
    if time_step == 0:
        return 0
    if len(misinformed_nodes) == 0:
        return 1
    recovery_rate = len(correct_nodes) / len(misinformed_nodes) / time_step
    return recovery_rate

def get_subgraph(graph, node_ids):
    """Get the subgraph of a graph containing the given node IDs.
    
    Parameters:
    graph: a networkx graph object
    node_ids: a list of node IDs
    
    Returns:
    subgraph: a networkx subgraph object
    """
    subgraph = graph.subgraph(node_ids)
    return subgraph

def compute_metrics(graph):
    """Compute a set of graph metrics.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    metrics: a dictionary of graph metrics
    """
    metrics = {
        'avg_path_length': compute_avg_path_length(graph),
        'clustering_coefficient': compute_clustering_coefficient(graph),
        'num_connected_components': compute_connected_components(graph),
        'diameter': compute_diameter(graph)
    }
    return metrics

def compute_metrics_non_scalar(graph):
    """Compute a set of non-scalar graph metrics.
    
    Parameters:
    graph: a networkx graph object
    
    Returns:
    metrics: a dictionary of non-scalar graph metrics
    """
    metrics = {
        'degree_distribution': compute_degree_distribution(graph)
    }
    return metrics

def compute_metrics_advanced(graph, correct_nodes, misinformed_nodes, source_node):
    "Compute a set of advanced graph metrics for information cascades."
    misinformed_subgraph = get_subgraph(graph, misinformed_nodes)
    correct_subgraph = get_subgraph(graph, correct_nodes)
    metrics = {
        'cascade_depth': compute_cascade_depth(correct_subgraph, source_node),
        'cascade_breadth': compute_cascade_breadth(graph, correct_subgraph),
        'structural_virality': compute_structural_virality(correct_subgraph),
        'fractional_resilience': compute_fractional_resilience(graph, correct_nodes),
        'topological_resilience': compute_topological_resilience(graph, misinformed_nodes),
        }
    return metrics

def compute_metrics_for_time_series(graph, df_time_series):
    """Compute metrics for a time series of data.
    
    Parameters:
    graph: a networkx graph object
    df_time_series: a pandas dataframe series containing a single time series
    
    Returns:
    metrics: a list of dictionaries containing metrics for each time step
    """
    source_node_id = df_time_series['source_node_id'].iloc[0]
    simulation_run_id = df_time_series['simulation_run_id'].iloc[0]
    correct_nodes = df_time_series['correct_agent_ids'].reset_index(drop=True)
    misinformed_nodes = df_time_series['misinformed_agent_ids'].reset_index(drop=True)
    rounds = df_time_series['round'].reset_index(drop=True)
    metrics = []
    for i in range(len(rounds)):
        correct_nodes_i = correct_nodes[i]
        misinformed_nodes_i = misinformed_nodes[i]
        round = rounds[i]
        metrics_i = compute_metrics_advanced(graph,
                                             correct_nodes_i,
                                             misinformed_nodes_i,
                                             source_node_id)
        metrics_i['round'] = round
        metrics_i['simulation_run_id'] = simulation_run_id
        metrics.append(metrics_i)
    return metrics

def compute_all_graph_metrics_from_model_data(model_data, graphs):
    """Compute all graph metrics from a dictionary of data.
    
    Parameters:
    model_data: a pandas DataFrame containing model data
    graphs: a dictionary of networkx graph objects, keyed by simulation_id
    
    Returns:
    all_metrics: a dictionary of all computed metrics
    """
    all_metrics = {
        'basic_metrics': [],
        'advanced_metrics': []
    }
    unique_simulation_run_ids = model_data['simulation_run_id'].unique()
    for simulation_run_id in unique_simulation_run_ids:
        graph = graphs[simulation_run_id]
        sim_data = model_data[model_data['simulation_run_id'] == simulation_run_id]

        # Compute basic graph metrics
        basic_metrics = compute_metrics(graph)
        basic_metrics["simulation_run_id"] = simulation_run_id
        assert src.utils.dict_values_are_scalar(basic_metrics)
        all_metrics['basic_metrics'].append(basic_metrics)

        # Compute advanced graph metrics for each simulation
        advanced_metrics = compute_metrics_for_time_series(graph, sim_data)
        for m in advanced_metrics:
            assert src.utils.dict_values_are_scalar(m)
        all_metrics['advanced_metrics'].extend(advanced_metrics)

    # Since we checked that all dict values are scalar, we can safely
    # convert the lists of dicts to pandas DataFrames
    all_metrics['basic_metrics'] = pandas.DataFrame(all_metrics['basic_metrics'])
    all_metrics['advanced_metrics'] = pandas.DataFrame(all_metrics['advanced_metrics'])

    return all_metrics
    

