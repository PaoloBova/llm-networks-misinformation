import networkx
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

@utils.method(init_graph)
def init_graph(unknown_type):
    raise Exception("Can't initialize a graph of this type")
