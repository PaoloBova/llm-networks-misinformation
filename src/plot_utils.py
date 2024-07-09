import matplotlib.pyplot as plt

def plot_simulation_results(params):
    """Plot the results of multiple simulations"""
    fig, ax = plt.subplots()  # Create a new figure and a new axes.
    
    num_agents = params['num_agents']
    model_df = params['results']['model']
    
    y_var = 'correct_count'
    
    # Group by simulation and round to calculate the mean result for each round
    grouped = model_df.groupby(['simulation', 'round'])[y_var].mean().reset_index()
    
    # Get the unique simulation ids
    simulation_ids = grouped['simulation'].unique()
    
    # Plot the results for each simulation
    for simulation_id in simulation_ids:
        simulation_data = grouped[grouped['simulation'] == simulation_id]
        ax.plot(simulation_data['round'],
                simulation_data[y_var]  / num_agents,
                marker='o',
                linestyle='-',
                color='lightcoral',
                alpha=0.5,
                label=f'Simulation {simulation_id}')
    
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