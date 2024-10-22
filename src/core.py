import autogen
import collections
import numpy as np
import pandas as pd
import random
import src.utils as utils
import src.data_utils as data_utils
from typing import Dict

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
    
    model.collect_stats(parameters)
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
    return model, agent_results, model_results

def get_autogen_chat_results(model, simulation_run_id):
    """Get the autogen chat results from the model and ensure they are in a JSON
    serializable format."""
    chat_results = {agent.name: agent.chat_messages
                    for agent in model.agents}
    
    # Chat messages are not JSON serializable, so build JSON serializable dicts
    # from them as follows:
    
    # TODO: Chat messages lack data on who sent each message. This makes it very
    # difficult to track who is who in the conversation. This should be fixed.
    
    chat_results = collections.defaultdict(list)
    for agent in model.agents:
        agent_key = str(agent.name)
        chat_messages = agent.chat_messages
        for peer_agent, chat in chat_messages.items():
            chat_id = f"agent1:{agent_key}_agent2:{str(peer_agent.name)}_sim:{simulation_run_id}"
            chat_results[chat_id].append(chat)
    return chat_results

def get_autogen_usage_summary(model):
    """Get the autogen usage summary from the model."""
    usage_summary = autogen.gather_usage_summary(model.agents)
    return usage_summary

def run_multiple_simulations(params:Dict, secrets:Dict={}) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations.
    secrets: A dictionary of secrets to be used in the simulations.
    
    Returns:
    A dictionary of DataFrames and objects which are JSON serializable.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    # If params is a dictionary, convert it into a list of dictionaries
    params_list = utils.dict_list(params) if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)
    print("Number of simulations: ", len(params_list))

    agent_results_all = []
    model_results_all = []
    chat_results_all = []
    usage_summaries_all = []
    graphs = {}
    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    simulation_id =  params_list[0]['simulation_id']
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == simulation_id
               for params in params_list)

    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    simulation_run_ids = [data_utils.create_id()
                          for _ in range(len(params_list))]
    for i, params in enumerate(params_list):
        set_random_seed(params['seed'])
        # We keep secrets separate from the rest of the params as we don't want
        # to expose them in the results
        args = {**params, **secrets}
        model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = simulation_run_ids[i]
        for res in agent_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
        usage_summaries_all.append(get_autogen_usage_summary(model))
        chat_results_all.append(get_autogen_chat_results(model, simulation_run_id))
        graphs[simulation_run_id] = model.graph
    
    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_all)
    model_df = pd.DataFrame(model_results_all)
    chat_data = {"usage_summaries": usage_summaries_all,
                 "chat_results": chat_results_all}
    
    # Return a dictionary of DataFrames and objects which are JSON serializable
    data = {'agent': agent_df,
            'model': model_df,
            **chat_data,
            "graphs": graphs,
            "params": [data_utils.filter_dict_for_json(params)
                       for params in params_list]}
    return data

def paramscan(params:Dict, secrets:Dict={}) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations.
    secrets: A dictionary of secrets to be used in the simulations.
    
    Returns:
    A dictionary of DataFrames and objects which are JSON serializable.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    # If params is a dictionary, convert it into a list of dictionaries
    params_list = utils.dict_list(params) if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)
    print("Number of simulations: ", len(params_list))

    agent_results_all = []
    model_results_all = []
    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    simulation_id =  params_list[0]['simulation_id']
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == simulation_id
               for params in params_list)

    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    simulation_run_ids = [data_utils.create_id()
                          for _ in range(len(params_list))]
    for i, params in enumerate(params_list):
        set_random_seed(params['seed'])
        # We keep secrets separate from the rest of the params as we don't want
        # to expose them in the results
        args = {**params, **secrets}
        _model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = simulation_run_ids[i]
        for res in agent_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
    
    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_all)
    model_df = pd.DataFrame(model_results_all)
    
    # Return a dictionary of DataFrames and objects which are JSON serializable
    data = {'agent': agent_df,
            'model': model_df,
            "params": [data_utils.filter_dict_for_json(params)
                       for params in params_list]}
    return data
