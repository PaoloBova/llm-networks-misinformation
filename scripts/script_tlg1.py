from dotenv import load_dotenv
import os
import random
from src.agent import NetworkAgent
import src.core
import src.data_utils
import src.metrics
from src.models import TechnologyLearningGame
import src.plot_utils
import src.prompts as prompts

# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
secrets = {"agent_secrets": {"spec_0": {"api_key": API_KEY},
                             "adjudicator": {"api_key": API_KEY}}}

# Simulation metadata
simulation_id = src.data_utils.create_id()
current_commit = src.data_utils.get_current_git_commit()

# Directories
data_dir = f"data/{simulation_id}"
plots_dir = f"plots/{simulation_id}"

# Setup logging
src.data_utils.setup_logging()

# Adjustable parameters
NUM_AGENTS = 8
NUM_ROUNDS = 8
TEMPERATURE = 0.2  # Adjust this to control the randomness of responses
prompt_functions = {"baseline_game": prompts.network_game}
agent_params = {"temperature": TEMPERATURE,
                "knowledge": {"decision": "A", "reasoning": "Initial state."},
                "knowledge_format": {"reasoning": str, "decision": int},
                "state": {"decision": 0,
                          "utility": 0,
                          "utility_gained": 0}}
agent_specs = [{"agent_spec_id": "spec_0",
                "agent_class": NetworkAgent,
                "num_agents": NUM_AGENTS,
                "agent_params": agent_params}]
adjudicator_spec = {"agent_spec_id": "adjudicator",
                    "agent_class": NetworkAgent,
                    "agent_params": {"temperature": TEMPERATURE}}
params = {"simulation_id": simulation_id,
          "commit": current_commit,
        #   "seed": [303, 606],
          "seed": [random.randint(0, 1000) for _ in range(5)],
        #   "network_seed": [639, 639],
          "network_seed": [random.randint(0, 1000) for _ in range(2)],
          "model_class": TechnologyLearningGame,
          "agent_specs": [agent_specs],
          "adjudicator_spec": adjudicator_spec,
          "num_rounds": NUM_ROUNDS,
          "prompt_functions": prompt_functions,
          "true_quality": 1,
          "prior_b_quality": "You have no prior on whether B is of high or low quality.",
          "hq_chance": 0.8,
          "src.networks.init_graph_type": "watts_strogatz_graph",
          "ws_graph_k": 2,
          "ws_graph_beta": 1,}

results = src.core.run_multiple_simulations(params, secrets=secrets)

results["seeds"] = [p["seed"] for p in results["params"]]
results["network_seeds"] = [p["network_seed"] for p in results["params"]]
src.data_utils.save_data(results, data_dir=data_dir)

# # It is safer to compute graph metrics after saving the simulation data.
# graph_metrics = src.metrics.compute_all_graph_metrics_from_model_data(results["model"], results["graphs"])
# src.data_utils.save_data(graph_metrics, data_dir=data_dir)

src.data_utils.save_data(results, data_dir=data_dir)

# Uncomment to draw the graphs
import matplotlib.pyplot as plt
import networkx
for graph in results["graphs"].values():
    networkx.draw(graph, with_labels=True)
    plt.show()

# plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})

# # Warning: save_name does not support all types of values
# simulation_stub = src.data_utils.save_name(params, sensitive_keys=["api_key"])

# plots = {f"simulation_results_{simulation_stub}": plot1}
# src.data_utils.save_plots(plots, plots_dir=plots_dir)
