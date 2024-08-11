import autogen
import collections
import numpy as np
import pandas as pd
import random
import src.utils as utils
import src.data_utils as data_utils

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

def get_autogen_chat_results(model):
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
            chat_id = f"agent1:{agent_key}_agent2:{str(peer_agent.name)}_sim:{model.simulation_id}"
            chat_results[chat_id].append(chat)
    return chat_results

def get_autogen_usage_summary(model):
    """Get the autogen usage summary from the model."""
    usage_summary = autogen.gather_usage_summary(model.agents)
    return usage_summary

def run_multiple_simulations(params):
    """Run multiple simulations and collect the results.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    # If params is a dictionary, convert it into a list of dictionaries
    params_list = utils.dict_list(params) if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)

    agent_results_all = []
    model_results_all = []
    chat_results_all = []
    usage_summaries_all = []
    graphs = {}
    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == params_list[0]['simulation_id']
               for params in params_list)
    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    simulation_run_ids = [data_utils.create_id()
                          for _ in range(len(params_list))]
    for i, params in enumerate(params_list):
        set_random_seed(params['seed'])
        model, agent_results, model_results = run_simulation(params)
        # Add a column to identify the simulation run and id
        # Simulation and model instantiation should not replace or modify simulation_id
        assert params['simulation_id'] == model.simulation_id
        simulation_run_id = simulation_run_ids[i]
        for res in agent_results:
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
        usage_summaries_all.append(get_autogen_usage_summary(model))
        chat_results_all.append(get_autogen_chat_results(model))
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
            "graphs": graphs}
    return data