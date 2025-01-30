import datetime
import hashlib
import json
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import networkx
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pprint
import random
import regex
import subprocess
from typing import Any, Dict, List, Union
import uuid
import logging

def setup_logging(log_file='chat_logs.log', level=logging.INFO):
    logging.basicConfig(filename=log_file, level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def is_plain_word(word):
    return word.isalpha()

def generate_random_phrase(words, num_words=3):
    return '_'.join(random.sample(words, num_words))

def create_id(path_to_data='/usr/share/dict/words', verbose=True):
    """Create a unique identifier based on a random phrase and a UUID.
    
    Parameters:
    - path_to_data: The path to a file containing a list of words.
    - verbose: Whether to print the random phrase and sim ID.
    
    Returns:
    - A unique identifier string.
    
    Note:
    - If the file at `path_to_data` does not exist, a UUID will be used instead.
    - You may wish to find a list of words to use as the dictionary file. On
    Unix systems, you can use `/usr/share/dict/words`. See the following link
    for more options: https://stackoverflow.com/questions/18834636/random-word-generator-python
    """
    try:
        with open(path_to_data, 'r') as f:
            words = [line.strip() for line in f if is_plain_word(line.strip())]
        sim_id = generate_random_phrase(words)
        if verbose:
            print(f"Random Phrase: {sim_id}")
        sim_id = f"{sim_id}_{str(uuid.uuid4())[:8]}"
        if verbose:
            print(f"Sim ID: {sim_id}")
        return sim_id
    except Exception as e:
        if verbose:
            print(f"You got exception {e}. Defaulting to a UUID.")
        sim_id = str(uuid.uuid4())
        if verbose:
            print(f"Sim ID: {sim_id}")
        return sim_id

def create_ids(num_ids):
    """
    Generate a given number of unique identifiers (UUIDs).

    Parameters:
    - num_ids: The number of unique identifiers to generate.

    Returns:
    - A list of unique identifier strings.
    
    Notes:
    - Use this function to efficiently generate multiple unique identifiers
      at once (and when you don't care about the ids being human-readable).
    """
    return [str(uuid.uuid4()) for _ in range(num_ids)]

def get_current_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit = "git command failed, are you in a git repository?"
    return commit

def display_chat_messages(filepath):
    # Load the chat histories from the JSON file
    with open(filepath, 'r') as f:
        chat_histories = json.load(f)

    # Iterate over the chat histories
    for chat_history in chat_histories:
        # Print the chat id
        for chat_id, chats in chat_history.items():
            print(f'## Chat ID: {chat_id}\n')
            # Iterate over the chats in the chat history
            for chat in chats:
                # Pretty print the chat
                pprint.pprint(chat)

def save_name(params,
              hash_filename=False,
              sensitive_keys=[], 
              max_depth=1,
              max_length=100):
    """Create a filename stub from a dictionary of parameters."""
    # Convert the dictionary to a list of strings
    param_list = []
    for k, v in sorted(params.items()):  # Sort items by key
        if k in sensitive_keys:
            continue

        if isinstance(k, str):
            if k == '':
                raise TypeError(f"Unsupported key type: empty string")
        elif isinstance(k, (int)):
            k = str(k)
        else:
            raise TypeError(f"Unsupported key type for key {k}: {type(k)}")

        if isinstance(v, (float, int)):
            # Round the float to 2 decimal places and replace the decimal point with 'p'
            v = str(round(v, 2)).replace('.', 'p').replace('-', 'minus')
        elif isinstance(v, bool):
            v = 1 if v else 0
        elif isinstance(v, (str, type(None))):
            v = str(v)
            if v == '':
                v = 'empty_string'
        elif callable(v):
            if hasattr(v, '__self__') and v.__self__ is not None:
                v = f"{v.__self__.__class__.__name__}.{v.__name__}"
            else:
                v = v.__name__
        elif isinstance(v, complex):
            v = str(v).replace('+', 'add').replace('-', 'minus')
        elif isinstance(v, datetime.datetime):
            v = v.isoformat()
        elif isinstance(v, uuid.UUID):
            v = str(v)
        elif isinstance(v, dict) and max_depth > 0:
            nested_stub = save_name(v,
                                    hash_filename,
                                    sensitive_keys,
                                    max_depth - 1,
                                    max_length // 2)
            v = f"begin_dict_{nested_stub}_end_dict"
        else:
            continue  # Skip unsupported types
        param_list.append(f"{k}_{v}")
    
    # Join the list into a single string with underscores
    param_str = "__".join(param_list)
    
    # Ensure filename is not too long
    if len(param_str) > max_length:
        param_str = param_str[:max_length]
    param_str = param_str[:255] # Bash has a hard limit of 255 characters for byte strings
    
    if len(param_str) == 0:
        raise ValueError("No parameters to save")
    if len(param_str) > 255:
        print(ValueError("Filename too long for bash. Consider excluding some variables if truncation is unacceptable."))
    
    # Hash the string to ensure uniqueness if hash_filename is True
    filename_stub = hashlib.md5(param_str.encode()).hexdigest() if hash_filename else param_str
    
    return filename_stub

def serialize_graphs(graphs):
    """Serialize a dictionary of NetworkX graphs to a JSON-serializable format."""
    serialized_graphs = {}
    for key, graph in graphs.items():
        # Convert node labels to integers
        G = networkx.convert_node_labels_to_integers(graph)

        # Convert attributes (e.g. block) to native Python int
        for node, data in G.nodes(data=True):
          if 'block' in data:
            data['block'] = int(data['block'])
        if 'partition' in G.graph:
            G.graph['partition'] = [list(s) for s in G.graph['partition']]

        serialized_graph = networkx.adjacency_data(G)
        serialized_graphs[key] = serialized_graph
    return serialized_graphs

def filter_dict_for_json(d):
    """
    Recursively filter out values from a dictionary that cannot be serialized to JSON.

    Parameters:
    d: a dictionary

    Returns:
    A new dictionary with only JSON serializable values.
    """
    filtered_dict = {}

    for key, value in d.items():
        if isinstance(value, dict):
            filtered_dict[key] = filter_dict_for_json(value)
        elif isinstance(value, list):
            filtered_list = []
            for item in value:
                if isinstance(item, dict):
                    filtered_list.append(filter_dict_for_json(item))
                else:
                    try:
                        json.dumps(item)
                        filtered_list.append(item)
                    except TypeError:
                        continue
            filtered_dict[key] = filtered_list
        else:
            try:
                json.dumps(value)
                filtered_dict[key] = value
            except TypeError:
                continue

    return filtered_dict

def append_data(data, file_path, format='csv'):
    if format == 'csv':
        file_exists = os.path.isfile(file_path)
        data.to_csv(file_path, mode='a', header=not file_exists, index=False)
    elif format == 'hdf5':
        with pd.HDFStore(file_path, mode='a') as store:
            store.append('my_key', data, format='table', data_columns=True)
    else:
        raise ValueError("Unsupported format")

def append_ndjson(record, file_path):
    with open(file_path, 'a') as f:
        f.write(json.dumps(record))
        f.write('\n')

def read_ndjson(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def save_data(data, data_dir=None, append=False):
    """
    Save each dataframe in `data` to the given folder. 
    
    Parameters:
    - data: A dictionary where the keys are the names of the files and the values are the dataframes to save.
    - data_dir: The directory to save the data in. If not provided, a random directory will be generated.
    - append: If True, append the data to the existing file. If False, overwrite the file.
    
    Usage:
    ```{python}
    data = {'df': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 'dict': {'key1': 'value1', 'key2': 'value2'}}
    save_data(data, 'my_folder')
    If no folder name is provided, generate a random one
    ```
    """
    if data_dir is None:
        data_dir = f'data/{create_id()}'
    # Create the folder if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    logging.info("Saving data to:", data_dir)

    # Iterate over the items in the dictionary
    for key, value in data.items():
        csv_path = os.path.join(data_dir, f'{key}.csv')
        json_path = os.path.join(data_dir, f'{key}.json')

        # DataFrame -> CSV
        if isinstance(value, pd.DataFrame):
            if append and os.path.isfile(csv_path):
                value.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                value.to_csv(csv_path, index=False)

        # If the value is a dictionary of networkx graphs, save it as a JSON file
        elif isinstance(value, dict) and all(isinstance(graph, networkx.Graph)
                                             for graph in value.values()):
            serialized_graphs = serialize_graphs(value)
            if append and os.path.isfile(json_path):
                append_ndjson(serialized_graphs, json_path)
            else:
                with open(json_path, 'w') as f:
                    json.dump(serialized_graphs, f)
        # If the value is a dictionary or list of dictionaries, save it as a
        # JSON file
        elif isinstance(value, (dict, list)):
            if append and os.path.isfile(json_path):
                append_ndjson(serialized_graphs, json_path)
            else:
                with open(json_path, 'w') as f:
                    json.dump(value, f)

def save_chunk(filepaths, chunk, filepath_key):
    """Save a single chunk of data to an HDF5 file, appending if the file exists."""
    filepath = os.path.join(filepaths['data_dir'], 
                            filepaths.get(filepath_key, filepath_key))
    filepath = f"{filepath}.hdf5"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    chunk_df = pd.DataFrame(chunk)
    
    with h5py.File(filepath, 'a') as f:
        for column in chunk_df.columns:
            data = chunk_df[column].values
            if column in f:
                # Append data to the existing dataset
                dataset = f[column]
                dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                dataset[-data.shape[0]:] = data
            else:
                maxshape = (None,)
                f.create_dataset(column,
                                 data=data,
                                 maxshape=maxshape, chunks=True)
    
    print(f"Saved chunk to {filepath}")

def save_inputs(filename, inputs, data_dir='data'):
    """Save the inputs to filepath if no such file exists."""
    inputs_dir = os.path.join(data_dir, 'inputs')
    filepath = os.path.join(inputs_dir, f'{filename}.json')
    inputs = [filter_dict_for_json(d) for d in inputs]
    if not os.path.exists(filepath):
        save_data({filename: inputs}, data_dir=inputs_dir)

def save_sim_to_tracker(data_dir: str, sim_id: str, batch_id: Union[str, None] = None):
    """
    Save simulation metadata to the tracker file.

    Args:
        data_dir (str): The directory where the tracker file is located.
        sim_id (str): The simulation ID.
        batch_id (str, optional): The batch ID. Defaults to None.
    """
    # Define the path to the simulation tracker file
    sim_tracker_path = os.path.join(data_dir, "sim_tracker.csv")

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Check if the simulation tracker file exists
    file_exists = os.path.isfile(sim_tracker_path)

    # Create a DataFrame with the simulation metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sim_metadata = pd.DataFrame({
        "sim_id": [sim_id],
        "batch_id": [batch_id],
        "timestamp": [timestamp]
    })

    # Append the simulation metadata to the tracker file, creating the file if it doesn't exist
    try:
        if file_exists:
            sim_metadata.to_csv(sim_tracker_path, mode='a', header=False, index=False)
        else:
            print("Creating a new simulation tracker file.")
            sim_metadata.to_csv(sim_tracker_path, mode='w', header=True, index=False)
        print(f"Simulation ID {sim_id} has been added to the tracker.")
    except Exception as e:
        print(f"An error occurred while updating the simulation tracker: {e}")

def save_plots(plots, plots_dir=None):
    """
    Save each plot in `plots` to the given folder.
    
    Parameters
    ----------
    plots : dict
        A dictionary where the keys are the plot names and the values are the
        plot objects.
    data_dir : str
        The directory to save the plots in. If not provided, a random directory
        will be generated.
    
    Usage:
    ```{python}
    save_plots({'plot1': fig1, 'plot2': fig2}, data_dir='my_folder')
    ```
    """
    if plots_dir is None:
        plots_dir = f'plots/{create_id()}'
    # Create the folder if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Iterate over the items in the dictionary
    for filename_stub, plot in plots.items():
        # If plot is a matplotlib figure, save it as a PNG file
        if isinstance(plot, plt.Figure):
            filepath = os.path.join(plots_dir, f'{filename_stub}.png')
            plot.savefig(filepath)
            print(f"Saved file: {filepath}")
        # If plot is a plotly figure, save it as an HTML file
        elif isinstance(plot, go.Figure):
            filepath = os.path.join(plots_dir, f'{filename_stub}.html')
            pio.write_html(plot, filepath)
            print(f"Saved file: {filepath}")
        elif isinstance(plot, FuncAnimation):
            # This is a matplotlib animation. Save it as a GIF file
            filepath = os.path.join(plots_dir, f'{filename_stub}.gif')
            plot.save(filepath, writer='imagemagick')
            print(f"Saved file: {filepath}")

def extract_data(message: str, data_format: Dict[str, type]) -> List[Dict[str, Any]]:
    """Extracts data from a message according to a specified format.

    Args:
        message (str): The message to extract data from.
        data_format (Dict[str, type]): A dictionary specifying the expected keys and their corresponding types in the data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the extracted data. Each dictionary has keys and values corresponding to the data_format argument.
    """
    # Use a regular expression to extract all JSON strings from the message.
    matches = regex.findall(r'\{(?:[^{}]|(?R))*\}', message)
    data = []
    for match in matches:
        # Parse the JSON string into a dictionary.
        try:
            new_data = json.loads(match)
        except json.JSONDecodeError:
            print("Could not parse a JSON string.")
            continue

        # Validate the keys in the dictionary.
        expected_keys = set(data_format.keys())
        if set(new_data.keys()) != expected_keys:
            print("Received unexpected keys.")
            continue

        # Validate the types of the values in the dictionary.
        valid_data = True
        for key, expected_type in data_format.items():
            if not isinstance(new_data[key], expected_type):
                print(f"Received data with incorrect type for key '{key}'.")
                valid_data = False
                break

        if valid_data:
            data.append(new_data)

    return data

def sanitize_dict_values(results_dict):
    """
    Sanitizes the values in the input dictionary.

    If a value is a numpy array, it is reshaped to 1D and resized to match the
    length of the longest array in the dictionary.

    If a value is not a numpy array, it is converted into a numpy array with the
    same length as the longest array in the dictionary.

    If the input dictionary is empty or None, a warning is printed and an empty
    dictionary is returned.

    Parameters:
    results_dict (dict): The input dictionary to sanitize.

    Returns:
    dict: The sanitized dictionary.
    """

    if not results_dict:
        print("Warning: Input dictionary is empty or None.")
        return {}
    
    max_length = max(len(v) for v in results_dict.values() if isinstance(v, np.ndarray))
    
    for key, value in results_dict.items():
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                print(f"Warning: {key} is not a 1D array. Reshaping to 1D.")
                value = value.ravel()
            if len(value) < max_length:
                value = np.resize(value, max_length)
        else:
            value = np.full(max_length, value)
        results_dict[key] = value

    return results_dict