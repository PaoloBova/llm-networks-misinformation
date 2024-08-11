import matplotlib.pyplot as plt

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

def plot_metric_against_var(data,
                            metric='fraction_correct_nodes',
                            var='rewiring_probability',
                            group_var=None,
                            data_key='results',
                            plot_type='line',
                            plotting_library='matplotlib',
                            marker='o',
                            linestyle='-',
                            color='lightcoral',
                            alpha=0.5,
                            title='Title',
                            xlabel='X Label',
                            ylabel='Y Label',
                            ylim=None,
                            xticks=None):
    """Plot `metric` aginst the given `topology_var` using the data in `data`.
    
    If relevant, plot a line for each group in the data.
    
    Parameters:
    data: a dictionary of dataframes or json data
    metric: the metric to plot
    var: the variable to plot against
    group_var: the variable to group the data by
    plot_type: the type of plot to create
    
    Returns:
    fig: a matplotlib figure object
    """
    
    fig, ax = plt.subplots()
    
    # Look at the relevant dataframe in the data dictionary
    df = data.get(data_key)
    if df is None:
        raise ValueError(f'No data found for key: {data_key}')
    
    if plotting_library == 'matplotlib':
        if plot_type == 'line':
            # Plot a line for each group in the data
            # If no group_var is provided, plot a single line
            if group_var is None:
                ax.plot(df[var], df[metric], marker=marker, linestyle=linestyle, color=color, alpha=alpha)
            else:
                for group in df[group_var].unique():
                    group_df = df[df[group_var] == group]
                    ax.plot(group_df[var], group_df[metric], marker=marker, linestyle=linestyle, color=color, alpha=alpha, label=group)
        else:
            raise ValueError(f'Plot type not supported: {plot_type}')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.legend()
        ax.grid(True)
        
    return fig
