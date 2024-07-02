# llm-networks-misinformation
Research and code exploring network structures to prevent misinformation from going viral in large language models (LLMs).

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/paolobova/llm-networks-misinformation.git
cd llm-networks-misinformation
pip install -r requirements.txt
```

To run our experiments, call a relevant module from scripts. For example:

```bash
python -m scripts.script1
```

## Project Structure

```
llm_networks_misinformation/
├── README.md
├── prompt_templates/
│   └── prompt1.txt
├── results/
│   ├── 3_random_words/chat_history.json
│   └── 3_random_words/model_results.csv
├── scripts/
│   └── script1.py
├── src/
│   ├── agent.py
│   ├── core.py
│   ├── debate_manager.py
│   ├── prompt_utils.py
│   └── prompts.py
```

To run our experiments, call a relevant module from scripts. For example (after navigating to the project directory and installing dependencies):

```bash
python -m scripts.script1
```

The above command will run the script1.py file.

Our example script script1.py has the following structure:

```python
import src.core
import src.models
import src.prompts
import src.plots
import networkx

seed = 1
graph = networkx.wattzstrogattz(...)
prompt_fns = {"baseline": src.prompts.prompt1}
llm_configs = [...]
model = src.models.debate_manager
inputs = {"n-agents": 4,
          "seed" : seed,
          "llm-prompt-fns": prompt_fns,
          "llm-configs": llm_configs,
          "graph": graph,
          "model": model,
          ...}

results = src.core.run_sim(inputs)
save_data(results)

plots = src.plots.plot_results(results)
save_plots(plots)
```

Prompt templates are stored in `prompts/` as `.txt` or `.md` files.

We create prompt functions in `prompts.py` that update placeholder text in the template given the current values of state variables in the simulation. This makes it easier to work on the prompt templates separately from the rest of the code. We write placeholder text in curly brackets `{example_var}`.

Agent definitions are stored in `agents.py`. We usually define classes which
inherit the `autogen.ConversableAgent` class so that we can easily work with
Autogen.

Our models are stored in `models.py`. Each model contains the
following components:
- A function for initializing agents, using the definitions from `agents.py`
- A step function that contains the logic for one time step of the simulation
- The logic for how agents communicate, often making use of Autogen.

Results are saved to a new subfolder in results/.
The new subfolder is named using 3 random words which
is also saved to the results as the sim_id variable. In
addition, results store the git commit and the date that
the simulation was run for. We split results across the
following files:
- `chat history.json`
- `model_stats.csv`


## Contributing
We welcome contributions to this project. Please see our CONTRIBUTING.md file for more information.

## License
This project is licensed under the terms of the MIT License.