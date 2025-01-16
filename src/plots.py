# Use this python file to write plotting functions to call in our scripts.
# Always name plotting functions after what they plot.
import math
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx
import numpy
import src.plot_utils
import src.utils as utils

def plot_metric_against_topology(data,
                                 metric='correct_count',
                                 var='round',
                                 title=None,
                                 group_var='simulation_run_id',
                                 data_key='model',):
    """Plot the metric against the topology using the data in data.
    
    Parameters:
    data: a dictionary of dataframes or json data
    metric: the metric to plot
    var: the variable to plot against
    
    Returns:
    fig: a matplotlib figure object
    """
    
    # Look at the relevant dataframe in the data dictionary
    df = data.get(data_key)
    if df is None:
        raise ValueError(f'No data found for key: {data_key}')
    
    fig = src.plot_utils.plot_metric_against_var(df,
                                            metric=metric,
                                            var=var,
                                            group_var=group_var,
                                            legend=None,
                                            plot_type='line',
                                            marker='o',
                                            linestyle='-',
                                            color='lightcoral',
                                            alpha=0.5,
                                            title=title,
                                            xlabel=var,
                                            ylabel='% Correct',
                                            ylim=None)
    
    return fig

def plot_royal_family_network(G):
    """
    Visualize a royal family network with different node types and edge colors.
    
    Parameters:
    - G: NetworkX Graph with nodes and edges.
    
    Returns:
    - Dictionary with positions, network, plot, and ax.
    
    This function:
        1. Positions nodes in a royal family network layout.
        2. Colors nodes and edges based on their type.
        3. Draws the network with different edge colors and opacities.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Compute royal family size
    # The royal family is simply the set of nodes with the highest degree.
    # Handle edge cases where there are no nodes, all nodes are in the royal family
    degrees = dict(G.degree())
    if len(degrees) == 0:
        raise ValueError("No nodes in the graph")
    nodes_high_degree = [node for node, degree in degrees.items()
                         if degree == max(degrees.values())]
    royal_family_size = len(nodes_high_degree)
    if royal_family_size == 0:
        raise ValueError("No royal family in the graph")
    if royal_family_size == len(G.nodes()):
        raise ValueError("All nodes are in the royal family")
    
    pos = {}
    non_royal_count = len(G.nodes()) - royal_family_size
    
    # Position royal family in center in a small circle
    royal_radius = 0.2
    for i in range(royal_family_size):
        angle = 2 * numpy.pi * i / royal_family_size
        pos[i] = (royal_radius * numpy.cos(angle), royal_radius * numpy.sin(angle))
    
    # Position half of non-royal nodes in middle circle
    middle_radius = 0.6
    half_non_royal = non_royal_count // 2
    for i in range(royal_family_size, royal_family_size + half_non_royal):
        angle = 2 * numpy.pi * (i - royal_family_size) / half_non_royal
        pos[i] = (middle_radius * numpy.cos(angle), middle_radius * numpy.sin(angle))
    
    # Position other half in outer circle
    outer_radius = 1.0
    for i in range(royal_family_size + half_non_royal, len(G.nodes())):
        angle = 2 * numpy.pi * (i - (royal_family_size + half_non_royal)) / (non_royal_count - half_non_royal)
        pos[i] = (outer_radius * numpy.cos(angle), outer_radius * numpy.sin(angle))
    
    # Draw edges with different colors and opacity for different types
    # Draw local neighborhood connections
    local_edges = [(u, v) for (u, v) in G.edges() 
                   if u >= royal_family_size and v >= royal_family_size]
    networkx.draw_networkx_edges(G, pos, edgelist=local_edges,
                           alpha=0.3, edge_color='blue')
    
    # Draw connections to royal family
    royal_edges = [(u, v) for (u, v) in G.edges() 
                   if ((u >= royal_family_size and v < royal_family_size)
                       or (u < royal_family_size and v >= royal_family_size))]
    networkx.draw_networkx_edges(G, pos, edgelist=royal_edges,
                           alpha=0.1, edge_color='red')
    
    # Draw royal family connections
    rf_edges = [(u, v) for (u, v) in G.edges() 
                if u < royal_family_size and v < royal_family_size]
    networkx.draw_networkx_edges(G, pos, edgelist=rf_edges, 
                          alpha=0.5, edge_color='purple', arrows=True, arrowsize=10)
    
    # Draw nodes
    networkx.draw_networkx_nodes(G, pos, 
                          nodelist=range(royal_family_size, len(G.nodes())),
                          node_color='lightblue',
                          node_size=100)
    
    networkx.draw_networkx_nodes(G, pos,
                          nodelist=range(royal_family_size),
                          node_color='red',
                          node_size=300)
    
    plt.title("Royal Family Network")
    plt.axis('equal')
    plt.axis('off')
    
    # Add legend
    plt.plot([], [], 'ro', markersize=10, label='Royal Family')
    plt.plot([], [], 'o', color='lightblue', markersize=7, label='Regular Nodes')
    plt.plot([], [], color='purple', alpha=0.5, label='Royal Family Connections')
    plt.plot([], [], color='red', alpha=0.3, label='Connections to Royal Family')
    plt.plot([], [], color='blue', alpha=0.3, label='Local Neighborhood Connections')
    plt.legend(loc='upper right')
    return {"pos": pos, "network": G, "plot": fig, "ax": ax}

def plot_sbm(G, ring_scale=5.0, sub_scale=2.0, node_size=100):
    """
    Plot a Stochastic Block Model network in matplotlib. Arranges nodes by block
    in a ring of circular clusters, color-coded by block.
    
    Parameters:
    - G: NetworkX Graph with a 'block' attribute on each node.
    - ring_scale: Factor controlling spacing between block clusters on the ring.
    - sub_scale: Factor controlling spacing within each block’s circular layout.
    - node_size: Size of each node for drawing.
    

    Returns:
    - Dictionary with positions, colors, network, plot, and ax.

    This function:
      1. Groups nodes by their 'block' attribute.
      2. Places each block in its own circular sub-layout around a ring, spaced by ring_scale.
      3. Color-codes nodes by block, omitting labels for clarity.
    """
    # Group nodes by block
    block_map = {}
    for node, data in G.nodes(data=True):
        block = data.get('block', 0)
        block_map.setdefault(block, []).append(node)

    # Calculate ring radius
    num_blocks = len(block_map)
    ring_radius = ring_scale * (len(G)**0.5)
    angle_step = 2 * math.pi / max(num_blocks, 1)

    # Determine positions for all nodes
    pos = {}
    blocks = sorted(block_map.keys())

    for i, block in enumerate(blocks):
        # Center for this block on the ring
        theta = i * angle_step
        center = (ring_radius * math.cos(theta), ring_radius * math.sin(theta))

        # Lay out the subgraph of this block in a small circle around 'center'
        sub_pos = networkx.circular_layout(
            G.subgraph(block_map[block]),
            center=center,
            scale=sub_scale
        )
        pos.update(sub_pos)

    # Color by block
    colors = [G.nodes[n].get('block', 0) for n in G.nodes()]

    # Draw the graph without labels, smaller nodes

    fig, ax = plt.subplots(figsize=(15, 15))
    networkx.draw(G, pos=pos, node_color=colors, cmap=plt.cm.Set3, 
            with_labels=False, node_size=node_size)
    return {"pos": pos, "colors": colors, "network": G, "plot": fig, "ax": ax}

def plot_network_default(G,
                         pos=None,
                         node_color='gray',
                         node_size=100,
                         edge_color='gray',
                         edge_alpha=0.5):
    """
    Plot a network with nodes and edges in matplotlib.

    Parameters:
    - G: networkx Graph. The network to plot.
    - pos: dict, optional. Node positions as a dictionary with node IDs as keys and positions as values.
    - node_color: str or list, optional. Node color. Can be a single color or a list of colors.
    - node_size: int, optional. Node size.
    - edge_color: str, optional. Edge color.
    - edge_alpha: float, optional. Edge transparency.

    Returns:
    - Dictionary with positions, plot, and ax.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    if pos is None:
        pos = networkx.spring_layout(G)
    networkx.draw(G, pos=pos, with_labels=False, node_size=node_size,
                  node_color=node_color, edge_color=edge_color, alpha=edge_alpha)
    return {"pos": pos, "plot": fig, "ax": ax}

@utils.multi
def plot_network(params, **kwargs):
    """
    Plot a network in matplotlib.

    Parameters:
    - params: dict. Contains the network graph and other parameters.
    - kwargs: dict. Additional keyword arguments.

    Returns:
    - Dictionary with positions, colors, network, plot, and ax.
    """
    return params.get("src.networks.init_graph_type", None)

@utils.method(plot_network, "stochastic_block_model")
def plot_network(params, **kwargs):
    G = params.get("graph")
    return plot_sbm(G, **kwargs)

@utils.method(plot_network, "royal_family_network")
def plot_network(params, **kwargs):
    G = params.get("graph")
    return plot_royal_family_network(G)

@utils.method(plot_network) # Default case
def plot_network(params, **kwargs):
    G = params.get("graph")
    return plot_network_default(G, **kwargs)

def animate_graph(graph_args, df, column_map={}, color_map={}, interval=1000):
    """
    Animate a network where node colors change over time.

    Parameters:
    - graph_args: dict. Contains a networkx graph and specifies the graph_type.
    - df: pandas DataFrame. Contains the data for frames, node-ids, and colors.
    - column_map: dict. Maps 'frame', 'node_id', and 'color' to the corresponding columns in df.
    - color_map: dict, optional. Maps values in the 'color' column to actual colors.
    - interval: delay between frames in milliseconds.

    This function uses FuncAnimation to generate an animated sequence of frames,
    each coloring nodes by their 0/1 action at time t.
    """

    # Compute plot primitives once:
    G = graph_args.get("graph")
    if G is None:
        return {}
    fig_data = plot_network(graph_args)
    pos = fig_data['pos']
    fig = fig_data['plot']
    ax = fig_data['ax']
    
    # Create a dictionary mapping frames to node colors
    frame_colors = {}
    frame_col_name = column_map.get('frame', 'frame')
    node_id_col_name = column_map.get('node_id', 'node_id')
    color_col_name = column_map.get('color', 'color')
    for frame, group in df.groupby(frame_col_name):
        frame_colors[frame] = {str(row[node_id_col_name]): color_map.get(row[color_col_name], 'gray')
                               for _, row in group.iterrows()}
    
    def update(frame):
        ax.clear()
        colors = [frame_colors[frame].get(str(n+1), 'gray') for n in G.nodes()]
        networkx.draw_networkx_edges(G, pos, ax=ax)
        networkx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=100)
        ax.set_title(f"Time step {frame}")

    ani = FuncAnimation(fig, update, frames=sorted(frame_colors.keys()), interval=interval, repeat=True)
    return ani
