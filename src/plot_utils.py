import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(params):
    """Plot the results of multiple simulations"""
    fig, ax = plt.subplots()  # Create a new figure and a new axes.
    
    num_agents = params['num_agents']
    model_df = params['results']['model']
    
    y_var = 'correct_count'
    
    # Group by simulation_run and round to calculate the mean result for each round
    grouped = model_df.groupby(['simulation_run', 'round'])[y_var].mean().reset_index()
    
    # Get the unique simulation_run ids
    simulation_ids = grouped['simulation_run'].unique()
    
    # Plot the results for each simulation_run
    for simulation_id in simulation_ids:
        simulation_data = grouped[grouped['simulation_run'] == simulation_id]
        ax.plot(simulation_data['round'],
                simulation_data[y_var]  / num_agents,
                marker='o',
                linestyle='-',
                color='lightcoral',
                alpha=0.5,
                label=f'simulation_run {simulation_id}')
    
    # Calculate and plot the mean result for each round across all simulations
    mean_results = grouped.groupby('round')[y_var].mean()
    ax.plot(mean_results.index, mean_results / num_agents, marker='o', linestyle='-', color='orangered', label='Mean Proportion')
    
    ax.set_title('Proportion of Agents with Correct Answer Over Rounds')
    ax.set_xlabel('Round')
    ax.set_ylabel('Proportion with Correct Answer')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(mean_results.index)
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_metric_against_var(df,
                            metric='fraction_correct_nodes',
                            var='rewiring_probability',
                            group_var=None,
                            plot_type='line',
                            plotting_library='matplotlib',
                            marker='o',
                            linestyle='-',
                            color='lightcoral',
                            mute_colors=False,
                            alpha=0.5,
                            title='Title',
                            xlabel='X Label',
                            ylabel='Y Label',
                            legend="auto",
                            ylim=None,
                            xticks=None):
    """Plot `metric` against the given `topology_var` using the data in `data`.
    
    If relevant, plot a line for each group in the data.
    
    Parameters:
    df: a pandas dataframe with all of the data we want to plot
    metric: the metric to plot
    var: the variable to plot against
    group_var: the variable to group the data by and determine the color of the lines
    plot_type: the type of plot to create
    
    Returns:
    fig: a matplotlib figure object
    """
    
    fig, ax = plt.subplots()
    
    if plotting_library == 'matplotlib':
        if plot_type == 'line':
            # Plot a line for each group in the data
            # If no group_var is provided, plot a single line
            if group_var is not None:
                unique_groups = df[group_var].unique()
                if mute_colors:
                    colors = ['lightcoral'] * len(unique_groups)
                else:
                    colors = cm.rainbow(np.linspace(0, 1, len(unique_groups)))
                for i, group in enumerate(unique_groups):
                    group_df = df[df[group_var] == group]
                    if len(unique_groups) > 10:
                        group_label= f"{group_var}_batch"
                    else:
                        group_label = group
                    # Sort group_df by var
                    group_df = group_df.sort_values(by=var)
                    ax.plot(group_df[var], group_df[metric], marker=marker, linestyle=linestyle, color=colors[i], alpha=alpha, label=group_label)
            else:
                ax.plot(df[var], df[metric], marker=marker, linestyle=linestyle, color=color, alpha=alpha)
        else:
            raise ValueError(f'Plot type not supported: {plot_type}')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if legend == "auto":
            if group_var is not None:
                ax.legend()
        elif legend is not None:
            ax.legend(legend)
        ax.grid(True)
        
    return fig