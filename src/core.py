import numpy as np
import pandas as pd
import random

def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility in simulations.

    Parameters:
    seed (int): The seed value to be used for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    # If other libraries are used, their seeds should be set here as well.
    # Note that Autogen only offers a seed cache which must be set
    # each time a new API call to an LLM is made.

def initialize_agents(params):
    """Initialize and return a list of agents"""
    names = ['num_agents', 'api_key', 'temperature']
    num_agents, api_key, temperature = [params[k] for k in names]
    agent_class = params['agent_class']
    # TODO: Generalize the agent initialization process
    return [agent_class(agent_id=i+1,
                        api_key=api_key,
                        temperature=temperature)
            for i in range(num_agents)]

def run(model, parameters):
    print(f"Starting simulation with {len(model.agents)} agents.")

    for _ in range(model.num_rounds):
        print(f"Round {model.tick} begins.")
        model.step(parameters)
        #TODO: Allow more flexibility in when to collect what.
        model.collect_stats(parameters)
        print(f"Round {model.tick} ends.")
        print("-" * 50)

    return model.agent_results, model.model_results
    
def run_simulation(params):
    """Initialize the agents and model, then run the model."""
    model_class = params['model_class']
    agents = initialize_agents(params)
    model = model_class(agents, params)
    agent_results, model_results = run(model, params)
    return agent_results, model_results

def run_multiple_simulations(params):
    """Run multiple simulations and collect the results."""
    num_simulations = params['num_simulations']
    agent_results_all = []
    model_results_all = []
    set_random_seed(params['seed'])
    params['simulation_id'] = params.get('simulation_id', 0)
    for i in range(num_simulations):
        params['simulation_id'] += 1
        agent_results, model_results = run_simulation(params)
        # Add a column to identify the simulation number
        for res in agent_results:
            res['simulation'] = i + 1
        for res in model_results:
            res['simulation'] = i + 1
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_all)
    model_df = pd.DataFrame(model_results_all)
    
    # Return a dictionary of DataFrames
    return {'agent': agent_df, 'model': model_df}