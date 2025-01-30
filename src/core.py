import autogen
import collections
import logging
import numpy as np
import pandas as pd
import random
import src.utils as utils
import src.data_utils as data_utils
import tqdm
from typing import Dict, List, Union

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

def init_agents(params):
    """Initialize and return a list of agents"""
    agent_specs = params['agent_specs']
    agent_secrets = params.get('agent_secrets', {})
    agents = []
    # Randomise order of agent specs to avoid any implicit ordering
    random.shuffle(agent_specs)
    for spec in agent_specs:
        agent_class = spec['agent_class']
        num_agents = spec.get('num_agents', 1)
        agent_params = spec.get('agent_params', {})
        spec_id = spec.get('agent_spec_id', None)
        secrets = agent_secrets.get(spec_id, {})
        
        for _ in range(num_agents):
            agent_id = len(agents) + 1
            agent = agent_class(agent_id=agent_id, **agent_params, **secrets)
            agents.append(agent)

    return agents

def init_adjudicator(params):
    """Initialize and return an adjudicator agent"""
    spec = params['adjudicator_spec']
    agent_secrets = params.get('agent_secrets', {})
    agent_class = spec['agent_class']
    agent_params = spec.get('agent_params', {})
    spec_id = spec.get('agent_spec_id', None)
    secrets = agent_secrets.get(spec_id, {})
    return agent_class(agent_id=f"adjudicator_{0}", **agent_params, **secrets)

def run(model, parameters):
    logging.info(f"Starting simulation with {len(model.agents)} agents.")
    
    model.collect_stats(parameters)
    for _ in range(model.num_rounds):
        logging.info(f"Round {model.tick} begins.")
        model.step(parameters)
        #TODO: Allow more flexibility in when to collect what.
        model.collect_stats(parameters)
        logging.info(f"Round {model.tick} ends.")
        logging.info("-" * 50)

    return model.agent_results, model.model_results
    
def run_simulation(params):
    """Initialize the agents and model, then run the model."""
    model_class = params['model_class']
    if params.get('adjudicator_spec'):
        params["adjudicator_agent"] = init_adjudicator(params)
    if params.get('agent_specs'):
        agents = init_agents(params)
    else:
        agents = None
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
    logging.info(f"Number of simulations: {len(params_list)}")

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
    params_list = [{**params, "simulation_run_id": data_utils.create_id()}
                   for params in params_list]
    for i, params in tqdm.tqdm(enumerate(params_list)):
        set_random_seed(params['seed'])
        # We keep secrets separate from the rest of the params as we don't want
        # to expose them in the results
        args = {**params, **secrets}
        model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = params["simulation_run_id"]
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

def sanitize_params(params:Union[Dict, List[Dict]]) -> List[Dict]:
    """Sanitize the parameters for the simulations and convert to list."""
    # If params is a dictionary, convert it into a list containing a dictionary
    params_list = [params] if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)
    logging.info("Number of simulations: ", len(params_list))

    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    simulation_id =  params_list[0]['simulation_id']
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == simulation_id
               for params in params_list)
    
    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    run_ids = data_utils.create_ids(len(params_list))
    i = 0
    for params, run_id in zip(params_list, run_ids):
        i += 1
        utils.set_nested_value(params, ['simulation_run_id'], run_id)
        utils.set_nested_value(params, ['simulation_run'], i + 1)
    return params_list

def sanitize_filepaths(filepaths, simulation_id):
    """Sanitize the filepaths for saving the results."""
    # Ensuer we have a filepath for saving the inputs
    if not 'inputs' in filepaths:
        filepaths['inputs'] = f"{simulation_id}"
    # Make sure we at least have filepaths for saving the agent and model results
    if not 'model_results' in filepaths:
        filepaths['model_results'] = "model_results"
    if not 'agent_results' in filepaths:
        filepaths['agent_results'] = "agent_results"
    return filepaths

def run_sims_online(params:Union[Dict, List[Dict]],
                    secrets:Dict={},
                    collect_as_vectors:bool=False) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
    collect_as_vectors: Whether the collected results are dicts of vectors or
      scalars. In the former case, we need to concatenate the vectors before we
      can construct a dataframe from the results.
    
    Returns:
    A dictionary of DataFrames and objects which are JSON serializable.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    params_list = sanitize_params(params)
    agent_results_all = []
    model_results_all = []
    simulation_id =  params_list[0]['simulation_id']
    for params in params_list:
        set_random_seed(params['seed'])
        # We keep secrets separate from the rest of the params as we don't want
        # to expose them in the results
        args = {**params, **secrets}
        _model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = params['simulation_run_id']
        simulation_run = params["simulation_run"]
        for res in agent_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = simulation_run
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = simulation_run
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
    
    if collect_as_vectors:
        agent_results_new = {}
        if agent_results_all:
            for k in agent_results_all[0].keys():
                agent_results_new[k] = np.hstack([d[k] for d in agent_results_all])
        agent_results_new = data_utils.sanitize_dict_values(agent_results_new)
        
        model_results_new = {}
        if model_results_all:
            for k in model_results_all[0].keys():
                model_results_new[k] = np.hstack([d[k] for d in model_results_all])
        model_results_new = data_utils.sanitize_dict_values(model_results_new)

    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_new)
    model_df = pd.DataFrame(model_results_new)
    
    # Return a dictionary of DataFrames and objects which are JSON serializable
    data = {'agent': agent_df,
            'model': model_df,
            "params": [data_utils.filter_dict_for_json(params)
                       for params in params_list]}
    return data

def run_sims_offline(params:Union[Dict, List[Dict]],
                     secrets:Dict,
                     filepaths:Dict) -> Dict:
    """Run multiple simulations, assuming the model saves results to disk.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
        We keep secrets separate from the rest of the params as we don't want
        to expose them in the results
    filepaths: A dictionary of filepaths to save data to.
    
    Returns:
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    params_list = sanitize_params(params)
    simulation_id = params_list[0]['simulation_id']
    filepaths = sanitize_filepaths(filepaths, simulation_id)
    data_utils.save_inputs(filepaths['inputs'], params_list)
    for params in params_list:
        set_random_seed(params['seed'])
        args = {**params, **secrets, "filepaths": filepaths}
        run_simulation(args)
    return filepaths

def paramscan(params:Union[Dict, List[Dict]],
              secrets:Dict={},
              filepaths:Dict={},
              collect_as_vectors:bool=False) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
    filepaths: A dictionary of filepaths to save data to.
    collect_as_vectors: Whether the collected results are dicts of vectors or
      scalars. In the former case, we need to concatenate the vectors before we
      can construct a dataframe from the results.
    
    Returns:
    Either
      - A dictionary of DataFrames and objects which are JSON serializable.
      - A dictionary of filepaths where the results are saved.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    if not filepaths:
        data = run_sims_online(params, secrets, collect_as_vectors)
        return data
    else:
        # If filepaths is nonempty, assume results data is
        # automatically saved to disk as the model runs.
        filepaths = run_sims_offline(params, secrets, filepaths)
        return filepaths
