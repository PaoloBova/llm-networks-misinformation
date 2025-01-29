import networkx
import numpy
import src.utils as utils

@utils.multi
def init_graph(params):
    return params.get("src.networks.init_graph_type", None)

@utils.method(init_graph, "watts_strogatz_graph")
def init_graph(params):
    names = ["num_agents", "ws_graph_k", "ws_graph_beta", "network_seed"]
    num_agents, k, beta, network_seed = [params[k] for k in names]
    graph = networkx.watts_strogatz_graph(num_agents, k, beta, seed=network_seed)
    return graph

@utils.method(init_graph, "stochastic_block_model")
def init_graph(params):
    """
    Initialize a stochastic block model (SBM) graph.

    Parameters:
    - num_agents (int): Total number of agents (nodes) in the graph.
    - sbm_sizes (list of int): Sizes of the blocks in the SBM.
    - sbm_p (float): Probability of edges within blocks.
    - sbm_q (float): Probability of edges between blocks.
    - network_seed (int): Seed for the random number generator.
    - ensure_connected (str): Method to ensure the graph is connected. Options are 'resample' or 'augment'.

    Returns:
    - graph (networkx.Graph): The generated SBM graph.
    """
    names = ["num_agents", "sbm_sizes", "sbm_p", "sbm_q", "network_seed", "ensure_connected"]
    num_agents, sizes, p, q, network_seed, ensure_connected = [params.get(k) for k in names]

    # Ensure sbm_sizes is a list of integers
    if not all(isinstance(size, int) for size in sizes):
        raise ValueError("All elements in sbm_sizes must be integers")

    # Filter and adjust sbm_sizes to ensure sum(sbm_sizes) == num_agents
    total_size = sum(sizes)
    if total_size < num_agents:
        sizes.append(num_agents - total_size)
    elif total_size > num_agents:
        raise ValueError("The sum of sbm_sizes exceeds num_agents")

    # Ensure sbm_p and sbm_q are floats
    p = float(p)  # Within block link probability
    q = float(q)  # Between block link probability

    # Create the link probabilities matrix
    link_probabilities = [[p if i == j else q for j in range(len(sizes))]
                          for i in range(len(sizes))]
    
    nodelist = list(range(total_size))

    # Generate the SBM graph
    graph = networkx.stochastic_block_model(sizes,
                                            link_probabilities,
                                            nodelist=nodelist,
                                            seed=network_seed)

    if ensure_connected == 'resample':
        # Resample until the graph is connected
        i = 1
        while not networkx.is_connected(graph):
            graph = networkx.stochastic_block_model(sizes,
                                                    link_probabilities,
                                                    nodelist=nodelist,
                                                    seed=network_seed + i)
            i += 1
    elif ensure_connected == 'augment':
        # Augment edges to ensure the graph is connected
        if not networkx.is_connected(graph):
            components = list(networkx.connected_components(graph))
            for i in range(len(components) - 1):
                u = next(iter(components[i]))
                v = next(iter(components[i + 1]))
                graph.add_edge(u, v)

    return graph

@utils.method(init_graph, "erdos_renyi_graph")
def init_graph(params):
    names = ["num_agents", "er_graph_p", "network_seed"]
    num_agents, p, network_seed = [params[k] for k in names]
    graph = networkx.erdos_renyi_graph(num_agents, p, seed=network_seed)
    return graph

def create_royal_family_network(n_total=40, royal_family_size=3, local_neighbors=2):
    """
    Create a Royal Family network with:
    - Core royal family (fully connected)
    - Non-royal nodes with connections to royal family
    - Non-royal nodes with local neighborhood connections
    """
    G = networkx.Graph()
    G.add_nodes_from(range(n_total))
    
    # Connect royal family members
    royal_family = list(range(royal_family_size))
    for i in royal_family:
        for j in royal_family:
            if i != j:
                G.add_edge(i, j)
    
    # Connect others to royal family and create local neighborhoods
    non_royal = list(range(royal_family_size, n_total))
    for i in non_royal:
        # Connect to all royal family members
        for rf_member in royal_family:
            G.add_edge(i, rf_member)
        
        # Create local neighborhood connections
        # Connect to local_neighbors/2 nodes on each side
        for j in range(1, (local_neighbors // 2) + 1):
            idx1 = (i - royal_family_size + j) % len(non_royal) + royal_family_size
            idx2 = (i - royal_family_size - j) % len(non_royal) + royal_family_size
            G.add_edge(i, idx1)
            G.add_edge(i, idx2)
    
    return G

@utils.method(init_graph, "royal_family_graph")
def init_graph(params):
    n_total = params.get("num_agents", 40)
    royal_family_size = params.get("royal_family_size", 3)
    local_neighbors = params.get("royal_family_local_neighbors", 2)
    graph = create_royal_family_network(n_total, royal_family_size, local_neighbors)
    return graph

@utils.method(init_graph)
def init_graph(unknown_type):
    raise Exception("Can't initialize a graph of this type")
