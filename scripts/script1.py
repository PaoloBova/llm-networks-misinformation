import os
from dotenv import load_dotenv

from src.agent import Agent
import src.core
import src.data_utils
import src.metrics
from src.models import DebateManager
import src.plot_utils
import src.prompts as prompts

# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
secrets = {"api_key": API_KEY}

# Simulation metadata and directories
simulation_id = src.data_utils.create_id()
current_commit = src.data_utils.get_current_git_commit()
data_dir = f"data/{simulation_id}"
plots_dir = f"plots/{simulation_id}"

# Adjustable parameters
NUM_AGENTS = 4
CORRECT_ANSWER = "42"
NUM_ROUNDS = 4
TEMPERATURE = 0.2  # Adjust this to control the randomness of responses
prompt_functions = {"baseline_game": prompts.baseline_game}
params = {"simulation_id": simulation_id,
          "commit": current_commit,
          "seed": [1],
          "network_seed": [1],
          "model_class": DebateManager,
          "agent_class": Agent,
          "num_agents": NUM_AGENTS,
          "correct_answer": CORRECT_ANSWER,
          "num_rounds": NUM_ROUNDS,
          "temperature": TEMPERATURE,
          "prompt_functions": prompt_functions,
          "src.networks.init_graph_type": "watts_strogatz_graph",
          "ws_graph_k": 2,
          "ws_graph_beta": 0,}

results = src.core.run_multiple_simulations(params, secrets=secrets)
src.data_utils.save_data(results, data_dir=data_dir)

# It is safer to compute graph metrics after saving the simulation data.
graph_metrics = src.metrics.compute_all_graph_metrics_from_model_data(results["model"], results["graphs"])
src.data_utils.save_data(graph_metrics, data_dir=data_dir)

# # Uncomment to draw the graphs
# import matplotlib.pyplot as plt
# import networkx
# for graph in results["graphs"].values():
#     networkx.draw(graph, with_labels=True)
#     plt.show()

plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})

# Warning: save_name does not support all types of values
simulation_stub = src.data_utils.save_name(params, sensitive_keys=["api_key"])

plots = {f"simulation_results_{simulation_stub}": plot1}
src.data_utils.save_plots(plots, plots_dir=plots_dir)
