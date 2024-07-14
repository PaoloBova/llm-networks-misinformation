import os
from dotenv import load_dotenv
# from core import main
import networkx

from src.agent import Agent
import src.core
import src.data_utils
from src.models import DebateManager
import src.plot_utils
import src.prompts as prompts

# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

current_commit = src.data_utils.get_current_git_commit()

# Adjustable parameters
NUM_AGENTS = 4
CORRECT_ANSWER = "42"
NUM_ROUNDS = 4
NUM_SIMULATIONS = 2
TEMPERATURE = 0.2  # Adjust this to control the randomness of responses

seed = 1
network_seed = seed
k = 2
beta = 0.1
graph = networkx.watts_strogatz_graph(NUM_AGENTS, k, beta, seed=network_seed)

# Uncomment to draw the graph before running the simulation.
# The plot must be closed before the simulation can run. 
# networkx.draw(graph, with_labels=True)
# plt.show()

prompt_functions = {"baseline_game": prompts.summary_game}

params = {"num_agents": NUM_AGENTS,
          "correct_answer": CORRECT_ANSWER,
          "target_variable": "community_funding",
          "num_rounds": NUM_ROUNDS,
          "num_simulations": NUM_SIMULATIONS,
          "temperature": TEMPERATURE,
          "api_key": API_KEY,
          "commit": current_commit,
          "prompt_functions": prompt_functions,
          "seed": seed,
          "model_class": DebateManager,
          "agent_class": Agent,
          "graph": graph}

# We create the simulation id before any random seeds are set
simulation_id = src.data_utils.create_id()
data_dir = f"data/{simulation_id}"
plots_dir = f"plots/{simulation_id}"

results = src.core.run_multiple_simulations(params)

src.data_utils.save_data(results, data_dir=data_dir)

plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})

# Warning: save_name does not support all types of values
simulation_stub = src.data_utils.save_name(params, sensitive_keys=["api_key"])

plots = {f"simulation_results_{simulation_stub}": plot1}
src.data_utils.save_plots(plots, plots_dir=plots_dir)
