from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # For saving animations
import os
from src.agent import NetworkAgent
import src.core
import src.data_utils
import src.metrics
from src.models import TechnologyLearningGame
import src.networks
import src.plot_utils
import src.plots
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

# Save sim to tracker
src.data_utils.save_sim_to_tracker("data", simulation_id)

# Setup logging
src.data_utils.setup_logging()

# Adjustable parameters
NUM_AGENTS = 40
NUM_ROUNDS = 12
TEMPERATURE = 0.2  # Adjust this to control the randomness of responses
NUM_BLOCKS = 8 # Only used for SBM
P_CORRECT = 0.7 # Probability of correct initial decision
prompt_functions = {"baseline_game": prompts.network_game2}
agent_params = {"temperature": TEMPERATURE,
                "knowledge": {"decision": 0, "reasoning": "Initial state."},
                "knowledge_format": {"reasoning": str,"decision": int},
                "state": {"decision": 0,
                          "decision_old": 0,
                          "prior_b_quality": "You have no prior on whether B is of high or low quality.",
                          "utility": 0,
                          "utility_gained": 0}}
agent_params2 = {**agent_params,
                 "knowledge": {"decision": 1, "reasoning": "Initial state."},
                 "state": {"decision": 1,
                           "decision_old": 1,
                           "prior_b_quality": "You have no prior on whether B is of high or low quality.",
                           "utility": 0,
                           "utility_gained": 0}}
agent_specs = [{"agent_spec_id": "spec_0",
                "agent_class": NetworkAgent,
                "num_agents": int(NUM_AGENTS * (1-P_CORRECT)),
                "agent_params": agent_params},
               {"agent_spec_id": "spec_0",
                "agent_class": NetworkAgent,
                "num_agents": int(NUM_AGENTS * P_CORRECT),
                "agent_params": agent_params2}]
adjudicator_spec = {"agent_spec_id": "adjudicator",
                    "agent_class": NetworkAgent,
                    "agent_params": {"temperature": TEMPERATURE}}
params = {"simulation_id": simulation_id,
          "commit": current_commit,
          # "seed": [303, 279],
          "seed": [303, 279, 843, 221, 919, 732],
          # "seed": [random.randint(0, 1000) for _ in range(1)],
          "network_seed": [639, 453, 432, 138],
          # "network_seed": [639, 453],
          # "network_seed": [random.randint(0, 1000) for _ in range(1)],
          "model_class": TechnologyLearningGame,
          "agent_specs": [agent_specs],
          "adjudicator_spec": adjudicator_spec,
          "num_agents": NUM_AGENTS,
          "num_rounds": NUM_ROUNDS,
          "prompt_functions": prompt_functions,
          "initial_share_correct": P_CORRECT,
          "compute_utilities_at_end": True,
          "src.networks.init_graph_type": "stochastic_block_model",
          "sbm_num_blocks": NUM_BLOCKS,
          "sbm_sizes": [[NUM_AGENTS // NUM_BLOCKS for _ in range(NUM_BLOCKS)]],
          "sbm_p": 0.8,
          "sbm_q": 0.01,
          "royal_family_size": 3,
          "royal_family_local_neighbors": 2,
          "er_graph_p": 0.1,
          "ensure_connected": "augment"}

# Run and save simulations
results = src.core.run_multiple_simulations(params, secrets=secrets)
src.data_utils.save_data(results, data_dir=data_dir)

# Create simulation results plots
plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})
plot2 = src.plots.plot_metric_against_topology(results,
                                               data_key="model",
                                               metric="consensus_score",
                                               var="round",
                                               title="Consensus score over time",)
plot3 = src.plots.plot_metric_against_topology(results,
                                               data_key="model",
                                               metric="switch_rate",
                                               var="round",
                                               title="Switch rate over time",)
# Create graph plots and animations
agent_df = results["agent"]
sim_run_graph_types = {p["simulation_run_id"]: p["src.networks.init_graph_type"]
                       for p in results["params"]}
graph_plots = {}
for sim_run_id, graph in results["graphs"].items():
    graph_type = sim_run_graph_types[sim_run_id]
    graph_args = {"graph": graph,
                  "src.networks.init_graph_type": graph_type}
    # Plot network
    plot = src.plots.plot_network(graph_args)["plot"]
    graph_plots[f"graph_{sim_run_id}"] = plot
    # Animate network
    agent_df_sub = agent_df[agent_df["simulation_run_id"] == sim_run_id]
    anim = src.plots.animate_graph(graph_args,
                                   agent_df_sub,
                                   column_map={'frame': 'round',
                                               'node_id': 'agent_id',
                                               'color': 'decision'},
                                   color_map={0: "blue", 1: "red"})
    graph_plots[f"graph_animation_{sim_run_id}"] = anim

# Save plots and animations
plots = {f"simulation_results_{simulation_id}_{current_commit}": plot1,
         f"consensus_score_{simulation_id}_{current_commit}": plot2,
         f"switch_rate_{simulation_id}_{current_commit}": plot3,
         **graph_plots}
src.data_utils.save_plots(plots, plots_dir=plots_dir)
