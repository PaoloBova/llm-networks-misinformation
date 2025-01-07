import autogen
from typing import Dict, List, Union
from src.agent import Agent
from src.prompt_utils import generate_prompt_from_template
import src.utils as utils

def prompt_fn_example(sender: autogen.ConversableAgent,
                      recipient: autogen.ConversableAgent,
                      context: Dict) -> Dict:
    guess = sender.knowledge['guess']
    reasoning = sender.knowledge['reasoning']
    own_guess = recipient.knowledge['guess']
    own_reasoning = recipient.knowledge['reasoning']
    json_format_string = """{"guess": int, "reasoning": str}"""
    return {
        "role": "system",
        "content": f"""You've received information that the number might be {guess} because '{reasoning}'. Recall that you currently believe {own_guess} because of the following reason: "{own_reasoning}". Consider whether you should update your beliefs. Give your new guess and reasoning in json format even if your answer is unchanged: {json_format_string}"""
    }
    
def prompt_fn_test(sender: autogen.ConversableAgent,
                   recipient: autogen.ConversableAgent,
                   context: Dict) -> Dict:
    tick = context.get("tick", 0)
    if tick==1:
        return {
            "role": "system",
            "content": f"""This is the first round of the game. The correct answer is 42. Respond with 'I understand'."""
        }
    return {
        "role": "system",
        "content": f"""Repeat the initial line of our conversation I told you"""
    }

# Read prompt template from file
with open("prompt_templates/prompt1.txt", 'r') as file:
    prompt_template1 = file.read()

def map_placeholders_baseline_game(sender: autogen.ConversableAgent,
                                   recipient: autogen.ConversableAgent,
                                   context: Dict) -> Dict:
    """Specify how to map placeholder text to runtime values"""
    return {"guess": sender.knowledge['guess'],
            "reasoning": sender.knowledge['reasoning'],
            "own_guess": recipient.knowledge['guess'],
            "own_reasoning": recipient.knowledge['reasoning'],
            "json_format_string": sender.knowledge_format}

def baseline_game(sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = map_placeholders_baseline_game(sender, recipient, context)
    prompt = generate_prompt_from_template(replacement_dict, prompt_template1)
    return {"role": role, "content": prompt}


# Read prompt template from file
with open("prompt_templates/prompt2.md", 'r') as file:
    prompt_template2 = file.read()

def map_placeholders_summary_game(sender: autogen.ConversableAgent,
                                   recipient: autogen.ConversableAgent,
                                   context: Dict) -> Dict:
    """Specify how to map placeholder text to runtime values"""
    return {"peer_guess": sender.knowledge['guess'],
            "peer_reasoning": sender.knowledge['reasoning'],
            "peer_name": sender.name,
            "own_guess": recipient.knowledge['guess'],
            "own_reasoning": recipient.knowledge['reasoning'],
            "own_name": recipient.name,
            "target_variable": context["target_variable"],
            "json_format_string": sender.knowledge_format}

def summary_game(sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = map_placeholders_summary_game(sender, recipient, context)
    prompt = generate_prompt_from_template(replacement_dict, prompt_template2)
    return {"role": role, "content": prompt}

# Read prompt template from file
with open("prompt_templates/network_game_1.md", 'r') as file:
    prompt_network_initial1 = file.read()

# Read prompt template from file
with open("prompt_templates/network_game_continue_1.md", 'r') as file:
    prompt_network_continue1 = file.read()

def network_game(_sender: autogen.ConversableAgent,
                 recipient: autogen.ConversableAgent,
                 context: Dict) -> Dict:
    role = context.get("role", "system")
    graph = context["graph"]
    agents = context["agents"]
    neighbour_ids = list(graph.neighbors(recipient.agent_id - 1))
    neighbours = [agent for agent in agents if agent.agent_id - 1 in neighbour_ids]
    # Neighbour decisions should be represented as as a string in the form:
    # Neighbour {agent_id}: {decision} -> {utility} Utility gained.
    neighbour_decisions_str = "\n".join(
        [f"Neighbour {neighbour.agent_id}: {'B' if neighbour.state['decision']==1 else 'A'} -> {neighbour.state['utility_gained']} Utility gained."
         for neighbour in neighbours])
    time = context["tick"]
    hq_chance = context.get("hq_chance", 1)
    prior_b_quality = context.get("prior_b_quality", "You have no prior on whether B is of high or low quality.")
    if time <= 1:
        prompt_template = prompt_network_initial1
        replacement_dict = {"prior_b_quality": prior_b_quality,
                            "hq_chance": hq_chance,
                            "lq_chance": 1 - hq_chance,
                            "json_format_string": recipient.knowledge_format}
    else:
        prompt_template = prompt_network_continue1
        replacement_dict = {"neighbour_experiences": neighbour_decisions_str,
                            "utility_gain": recipient.state["utility_gained"],
                            "agent_id": recipient.agent_id}
    prompt = generate_prompt_from_template(replacement_dict, prompt_template)
    return {"role": role, "content": prompt}