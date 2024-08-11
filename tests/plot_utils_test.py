import pandas as pd
import numpy as np

from src.plot_utils import plot_metric_against_var

def test_plot_matric_against_var():
    
    # Create a DataFrame with fake data
    model_df = pd.DataFrame({
        'simulation': np.repeat(np.arange(10), 100),  # 10 simulations
        'round': np.tile(np.arange(100), 10),  # 100 rounds per simulation
        'correct_count': np.random.rand(1000)  # Random data for correct_count
    })

    # Decide other arguments
    y_var = 'correct_count'

    # Group by simulation and round to calculate the mean result for each round
    grouped = model_df.groupby(['simulation', 'round'])[y_var].mean().reset_index()

    # Calculate the mean result for each round across all simulations
    mean_results = grouped.groupby('round')[y_var].mean()

    # Add the mean results to the grouped dataframe
    grouped['mean'] = mean_results

    # Call the second function
    # Note: Usually, you don't need to specify as many arguments as I do here.
    plot_metric_against_var(data={'results': grouped},
                            metric=y_var,
                            var='round',
                            group_var='simulation',
                            data_key='results',
                            plot_type='line',
                            marker='o',
                            linestyle='-',
                            color='lightcoral',
                            alpha=0.5,
                            title='Proportion of Agents with Correct Answer Over Rounds',
                            xlabel='Round',
                            ylabel='Proportion with Correct Answer',
                            ylim=(0, 1.1))
    
    assert True # Checks that the function runs without error