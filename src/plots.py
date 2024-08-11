# Use this python file to write plotting functions to call in our scripts.
# Always name plotting functions after what they plot.
import matplotlib.pyplot as plt
import src.plot_utils

def plot_metric_against_topology(data,
                                 metric='correct_count',
                                 var='rewiring_probability'):
    """Plot the metric against the topology using the data in data.
    
    Parameters:
    data: a dictionary of dataframes or json data
    metric: the metric to plot
    var: the variable to plot against
    
    Returns:
    fig: a matplotlib figure object
    """
    
    fig = src.plot_utils.plot_metric_against_var(data=data,
                                            metric=metric,
                                            var=var,
                                            group_var='simulation',
                                            data_key='results',
                                            plot_type='line',
                                            marker='o',
                                            linestyle='-',
                                            color='lightcoral',
                                            alpha=0.5,
                                            title='Proportion of Agents with Correct Answer',
                                            xlabel=var,
                                            ylabel='% Correct',
                                            ylim=(0, 1.1))
    
    return fig