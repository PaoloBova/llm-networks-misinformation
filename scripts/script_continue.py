import json
import networkx
import pandas
import src.data_utils
import src.metrics
import src.plot_utils
import src.plots

## Load an existing data directory
data_dir = "data/confiscate_cancel_timetable_8acccf35"
plots_dir = "plots/confiscate_cancel_timetable_8acccf35"

# Load the data
results = {}
for filename in ["agent", "model", "basic_metrics", "advanced_metrics"]:
    with open(f"{data_dir}/{filename}.csv", "r") as f:
        # load csv data using the pandas module
        results[filename] = pandas.read_csv(f)
for filename in ["graphs", "params", "chat_results", "usage_summaries"]:
    with open(f"{data_dir}/{filename}.json", "r") as f:
        # load json data using the json module
        results[filename] = json.load(f)

# Convert the json graphs to networkx graphs
results["graphs"] = {k: networkx.node_link_graph(v)
                     for k, v in results["graphs"].items()}
# Convert model['correct_agent_ids'] to a series of vectors of ints
results["model"]["correct_agent_ids"] = results["model"]["correct_agent_ids"].apply(json.loads)

params = results["params"][0]

# Resume computations and plots

graph_metrics = src.metrics.compute_all_graph_metrics_from_model_data(results["model"], results["graphs"])
src.data_utils.save_data(graph_metrics, data_dir=data_dir)

# # Uncomment to draw the graphs
# import matplotlib.pyplot as plt
# import networkx
# for graph in results["graphs"].values():
#     networkx.draw(graph, with_labels=True)
#     plt.show()

plot1 = src.plots.plot_metric_against_topology(results,
                                               metric='correct_proportion',
                                               var='round',
                                               group_var='simulation_run_id',
                                               title='Proportion of Correct Agents Over Rounds',
                                               data_key='model')

# merge basic metrics and advanced metrics df with model df
combined_df = results['model'].merge(results['basic_metrics'], on='simulation_run_id')
# Create a dataframe containing the params values
params_list = results["params"]
params_df = pandas.DataFrame(params_list)
params_df["simulation_run"] = params_df.index
combined_df = combined_df.merge(params_df, on='simulation_run')
# Keep in mind that we need to match simulation_run_id and round vars for the advanced_metrics
combined_df = combined_df.merge(results['advanced_metrics'], on=['simulation_run_id', 'round'])

combined_df = combined_df.groupby(['ws_graph_beta', 'round'])['structural_virality'].mean().reset_index()

plot2 = src.plot_utils.plot_metric_against_var(combined_df,
                                               metric='structural_virality',
                                               var='round',
                                               group_var='ws_graph_beta',
                                               legend="auto",
                                               title='Structural Virality Against Rewiring Probability',)
# plot1 = src.plot_utils.plot_simulation_results({**params, "results": results})

# Warning: save_name does not support all types of values
# simulation_stub = src.data_utils.save_name(params)
simulation_stub = "fake_stub"
plots = {f"simulation_results_{simulation_stub}": plot1,
         f"structural_virality_{simulation_stub}": plot2}
src.data_utils.save_plots(plots, plots_dir=plots_dir)
print([m["usage_including_cached_inference"]['gpt-3.5-turbo-0125'] for m in results["usage_summaries"]])
usage_df = pandas.DataFrame([m["usage_including_cached_inference"]["gpt-3.5-turbo-0125"] for m in results["usage_summaries"]])
usage_df["total_cost"] = [m["usage_including_cached_inference"]["total_cost"] for m in results["usage_summaries"]]
# usage_df_excluding_cache = pandas.DataFrame([m["usage_excluding_cached_inference"]["gpt-3.5-turbo-0125"] for m in results["usage_summaries"]])
# usage_df_excluding_cache["total_cost"] = [m["usage_excluding_cached_inference"]["total_cost"] for m in results["usage_summaries"]]
print(usage_df)
print("Total Cost of Simulation (including cached inference): ", usage_df.total_cost.sum())
print("Total tokens used in simulation (including cached inference): ", usage_df.total_tokens.sum())
chat_turns_per_round = 1
n_agents = 8
n_rounds = 10
n_simulation_runs = len(params_list)
total_conversations = n_agents * n_rounds * n_simulation_runs * chat_turns_per_round
print(f"Avg token per turn in a conversation: {usage_df.total_tokens.sum() / total_conversations}")
# print("Total Cost of Simulation (excluding cached inference): ", usage_df_excluding_cache.total_cost.sum())
# print("Total tokens used in simulation (excluding cached inference): ", usage_df_excluding_cache.total_tokens.sum())