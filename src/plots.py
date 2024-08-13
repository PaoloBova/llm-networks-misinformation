# Use this python file to write plotting functions to call in our scripts.
# Always name plotting functions after what they plot.
import matplotlib.pyplot as plt
import src.plot_utils

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