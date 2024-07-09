import os
from dotenv import load_dotenv
# from core import main
import networkx

import src.prompts as prompts
from src.core import main

# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

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

prompt_functions = {"baseline_game": prompts.baseline_game}

params = {"num_agents": NUM_AGENTS,
          "correct_answer": CORRECT_ANSWER,
          "num_rounds": NUM_ROUNDS,
          "num_simulations": NUM_SIMULATIONS,
          "temperature": TEMPERATURE,
          "api_key": API_KEY,
          "prompt_functions": prompt_functions,
          "seed": seed,
          "graph": graph}

all_results = main(params)