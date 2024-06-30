from src.agent import Agent
from src.debate_manager import DebateManager
import matplotlib.pyplot as plt
import numpy as np

def initialize_agents(num_agents, api_key, temperature):
    # Initialize and return a list of Agent objects
    return [Agent(agent_id=i+1, api_key=api_key, temperature=temperature)
            for i in range(num_agents)]

def run_simulation(params):
    names = ['num_agents', 'correct_answer', 'num_rounds', 'api_key', 'temperature']
    num_agents, correct_answer, num_rounds, api_key, temperature = [params[k] for k in names]
    # Initialize agents and the debate manager, then start the debate rounds
    agents = initialize_agents(num_agents, api_key, temperature)
    debate_manager = DebateManager(agents, correct_answer, num_rounds)
    correct_counts_over_time = debate_manager.run(params)
    return correct_counts_over_time

def run_multiple_simulations(params):
    num_simulations = params['num_simulations']
    # Run multiple simulations and collect the results
    all_results = []
    for _ in range(num_simulations):
        result = run_simulation(params)
        all_results.append(result)
    return all_results

def plot_simulation_results(all_results, params):
    num_agents = params['num_agents']
    # Plot the results of multiple simulations
    rounds = np.arange(len(all_results[0]) + 1)
    adjusted_results = [np.insert(result, 0, 1) for result in all_results]
    
    for result in adjusted_results:
        plt.plot(rounds, np.array(result) / num_agents, marker='o', linestyle='-', color='lightcoral', alpha=0.5)
    
    mean_results = np.mean(adjusted_results, axis=0) / num_agents
    plt.plot(rounds, mean_results, marker='o', linestyle='-', color='orangered', label='Mean Proportion')
    
    plt.title('Proportion of Agents with Correct Answer Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Proportion with Correct Answer')
    plt.ylim(0, 1.1)
    plt.xticks(rounds)
    plt.legend()
    plt.grid(True)
    plt.show()

def main(params):
    all_results = run_multiple_simulations(params)
    plot_simulation_results(all_results, params)
    return all_results