import datetime
import hashlib
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pprint
import random
import subprocess
import uuid


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

def save_data(data, data_dir=None):
    """
    Save each dataframe in `data` to the given folder. 
    
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

    # Iterate over the items in the dictionary
    for key, value in data.items():
        # If the value is a DataFrame, save it as a CSV file
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(data_dir, f'{key}.csv'), index=False)
        # If the value is a dictionary or list of dictionaries, save it as a
        # JSON file
        elif isinstance(value, (dict, list)):
            with open(os.path.join(data_dir, f'{key}.json'), 'w') as f:
                json.dump(value, f)

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