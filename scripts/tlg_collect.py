import src.data_utils as data_utils
import src.plots

simulation_id, current_commit, data_dir, plots_dir = data_utils.setup_project(save_tracker=False)

combined_results = data_utils.load_simulation_results(f"data/sim_tracker.csv",
                                                      filenames=["agent.csv",
                                                                 "params.json",
                                                                 "model.csv"])

all_combined_results = data_utils.load_all_simulation_results("data",
                                                    filenames=["agent.csv",
                                                                  "params.json",
                                                                  "model.csv"])

print(all_combined_results)

print(combined_results)

# combined_results = all_combined_results

agent_data = combined_results["agent.csv"]

print(combined_results["params.json"].columns)
print(combined_results["model.csv"].columns)
import pandas as pd

# Get the dataframes from your combined results.
model_data = combined_results["model.csv"]
params_data = combined_results["params.json"]

# We need to filter out the following data:
# 1. Those for which num_agents is not 40
# 2. Those for which num_rounds is params_data 12

# Filter out the data for which num_agents is not 40
# model_data = model_data[model_data["num_agents"] == 40]
params_data = params_data[params_data["num_agents"] == 40]

# Filter out the data for which num_rounds is not 12
# model_data = model_data[model_data["num_rounds"] == 12]
params_data = params_data[params_data["num_rounds"] == 12]

# Identify all columns in params_data that vary in value across simulation runs.
varying_cols = []
varying_vals = {}
for col in params_data.columns:
    if col in ["simulation_id", "simulation_run_id", "sim_id", "seed", "network_seed",
               ]:
        continue
    try:
        # Convert each value to a string to handle unhashable types
        unique_vals = params_data[col].apply(lambda x: str(x)).unique()
        if len(unique_vals) > 1:
            varying_cols.append(col)
            varying_vals[col] = unique_vals
    except Exception as e:
        # If there's any error, skip this column
        continue

print(f"Varying columns: {varying_cols}")
print(f"Varying values: {varying_vals}")


# Find all differences in varying_vals["agent_specs"]
def find_differences(dict_list):
    """
    Recursively find differences in a list of dictionaries.
    
    For each key found in any of the dictionaries, this function checks if the corresponding values
    differ across the dictionaries. If the values are themselves dictionaries, the function recurses.
    Only keys with differences are returned.

    Parameters:
        dict_list (list of dict): The list of dictionaries to compare.

    Returns:
        dict: A dictionary mapping keys to either:
              - A list of differing values, or
              - A nested dictionary for differences in sub-dictionaries.
    """
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())

    differences = {}
    for key in all_keys:
        # Collect values for this key (including None when missing)
        values = [d.get(key) for d in dict_list]
        
        # If all non-None values are dictionaries, then do a recursive diff
        if all(isinstance(v, dict) for v in values if v is not None):
            # If some dictionaries are missing this key, that's a difference.
            if any(v is None for v in values):
                differences[key] = values
            else:
                nested_diff = find_differences(values)
                if nested_diff:  # Only record if there are differences in nested dicts.
                    differences[key] = nested_diff
        else:
            # For non-dict items, compare by creating a set of string representations
            # (this is not perfect, but works for many use-cases)
            unique_vals = {repr(v) for v in values}
            if len(unique_vals) > 1:
                # Also return the actual values (using a set to remove duplicates)
                differences[key] = list({v for v in values})
    return differences

agent_specs_unique = varying_vals["agent_specs"]
# convert strings in array to dictionaries
agent_specs_unique = [eval(x) for x in agent_specs_unique]
# Flatten list of lists
agent_specs_unique = [item for sublist in agent_specs_unique for item in sublist]
print("Agent specs unique:", agent_specs_unique)
diff = find_differences(agent_specs_unique)
print("Differences in agent_specs:", diff)

# Looks like we changed our mind about how to encode decisions in the agent state
# and knowledge. We should go with whatever is the newest format and discard the rest.
# In the newest format, decision, decision_old must match and be the same as
# the decision in the knowledge dictionary.

import json

def conforms_to_new_standard(agent_spec):
    """
    Check if the given agent_spec conforms to the new standard.
    The new standard requires that:
      - 'decision' and 'decision_old' exist, and 
      - both equal the value of knowledge["decision"]

    Parameters:
        agent_spec (dict or str): agent_spec as a dictionary or a JSON string.

    Returns:
        bool: True if the agent_spec conforms to the new standard, False otherwise.
    """
    # If agent_spec is a string, try to decode it
    if isinstance(agent_spec, str):
        try:
            agent_spec = json.loads(agent_spec)
        except Exception:
            return False

    if not isinstance(agent_spec, dict):
        return False

    agent_state = agent_spec.get("state")
    knowledge = agent_spec.get("knowledge")
    decision = agent_state.get("decision")
    decision_old = agent_state.get("decision_old")
    knowledge_decision = knowledge.get("decision")

    return decision == decision_old == knowledge_decision

print("Params data before filtering:", params_data)
# Assuming params_data is a pandas DataFrame that contains an 'agent_spec' column:
params_data = params_data[params_data["agent_specs"].apply(conforms_to_new_standard)]

print("Params data after filtering:", params_data)

# The following variables vary across simulation runs:
# ['seed', 'network_seed', 'sbm_p', 'sbm_q']


# Identify columns that are common to both dataframes (excluding the merge key)
common_cols = set(model_data.columns).intersection(set(params_data.columns)) - {"simulation_id"}

# Drop these duplicate columns from the params_data so that only model_data's versions remain.
filtered_params = params_data.drop(columns=list(common_cols))

# Now merge using the filtered params_data
merged_df = pd.merge(model_data, filtered_params, on="simulation_id", how="left")

print(merged_df)

# Now group the merged dataframe by the 'network_type' column. 
# We assume that the merged dataframe has a column named network_type (from src.networks.network_type).
grouped = merged_df.groupby("src.networks.init_graph_type")

# Display the size of each group
print(grouped.size())

results = {"sbm": grouped.get_group("stochastic_block_model"),
           "erdos_renyi": grouped.get_group("erdos_renyi_graph")}

print(results["sbm"].columns)

plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})
plot2 = src.plots.plot_metric_against_topology(results,
                                           data_key="sbm",
                                           metric="consensus_score",
                                           var="round",
                                           group_var="simulation_run_id",
                                           title="Consensus score over time",)
plot3 = src.plots.plot_metric_against_topology(results,
                                               data_key="sbm",
                                               metric="switch_rate",
                                               var="round",
                                               group_var="simulation_run_id",
                                               title="Switch rate over time",)

# Save plots and animations
plots = {f"consensus_score_{simulation_id}_{current_commit}": plot2,
         f"switch_rate_{simulation_id}_{current_commit}": plot3}
data_utils.save_plots(plots, plots_dir=f"extra_plots/{simulation_id}")

