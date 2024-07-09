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

## User Guide: Creating a New Script

This guide will walk you through the process of creating a new script in this project. The following is a basic structure of a script:


```python
import src.prompts
import src.plots
import networkx

# Set the seed for reproducibility
seed = 1

# Create a graph for the simulation
graph = networkx.wattzstrogattz(...)

# Define the prompt functions to be used
prompt_fns = {"baseline": src.prompts.prompt1}

# Define the LLM configurations
llm_configs = [...]

# Define the model to be used
model = src.models.debate_manager

# Define the inputs for the simulation
inputs = {
    "n-agents": 4,
    "seed" : seed,
    "llm-prompt-fns": prompt_fns,
    "llm-configs": llm_configs,
    "graph": graph,
    "model": model,
    ...
}

# Run the simulation
results = src.core.run_sim(inputs)

# Save the results
src.data_utils.save_data(results)

# Plot the results
plots = src.plots.plot_results(results)

# Save the plots
src.plot_utils.save_plots(plots)
```

### Step-by-step Guide

1. **Import necessary modules**: Import the necessary modules at the beginning of your script. This typically includes `src.prompts`, `src.plots`, `networkx`, and others as needed.

2. **Set up parameters**: Set up a dictionary of parameters that will be passed to the `src.core.run_sim` function. This includes the number of agents, the seed for reproducibility, the prompt functions, the LLM configurations, the graph, and the model.

3. **Run simulations**: Call `src.core.run_sim` with the parameters dictionary to run the simulations.

4. **Save and plot results**: Save the results using `src.data_utils.save_data` and plot the results using `src.plots.plot_results`. Then, save the plots using `src.plot_utils.save_plots`.

### Prompt Functions

Prompt functions are stored in `src.prompts`. These functions update placeholder text in the template given the current values of state variables in the simulation. This makes it easier to work on the prompt templates separately from the rest of the code. You can create new prompt functions or use existing ones.

To create your own prompt function, you need to define a function that takes three parameters: sender, recipient, and context. This function should return a dictionary with the keys role and content. 

```python
def my_prompt_fn(sender: autogen.ConversableAgent,
                 recipient: autogen.ConversableAgent,
                 context: Dict) -> Dict:
    # Extract necessary information from sender and recipient
    sender_info = sender.knowledge['info']
    recipient_info = recipient.knowledge['info']

    # Construct the content of the prompt
    content = f"Sender says: {sender_info}. Recipient thinks: {recipient_info}."

    # Return the prompt dictionary
    return {
        "role": "system",
        "content": content
    }
```


In this example, sender and recipient are instances of `autogen.ConversableAgent` and context is a dictionary that contains any additional information needed to generate the prompt. The function constructs a string content based on the knowledge of the sender and recipient, and returns a dictionary with role set to "system" and content set to the constructed string.

You can replace the sender_info and recipient_info extraction and the content construction with whatever logic is appropriate for your specific use case.

Once you've defined your prompt function, you can add it to the `prompt_fns` dictionary in your script to use it in a simulation. For example:


```python
prompt_fns = {"my_prompt": my_prompt_fn}
```

### Environment Variables

If your script uses environment variables, such as the OpenAI API key, make sure to set them up using a `.env` file and the `dotenv` package.

## Contributing
We welcome contributions to this project. Please see our CONTRIBUTING.md file for more information.

## License
This project is licensed under the terms of the MIT License.