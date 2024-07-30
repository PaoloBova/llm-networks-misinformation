import autogen
from typing import Dict, List, Union
from src.agent import Agent
from src.prompt_utils import generate_prompt_from_template
import src.utils as utils

@utils.multi
def build_prompt(_sender, _recipient, context):
    return context.get("prompt_name")

@utils.method(build_prompt, "baseline_game")
def build_prompt(sender, recipient, context):
    return baseline_game(sender, recipient, context)

@utils.method(build_prompt, "summary_game")
def build_prompt(sender, recipient, context):
    return summary_game(sender, recipient, context)

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
    return {"guess": sender.knowledge['guess'],
            "reasoning": sender.knowledge['reasoning'],
            "own_guess": recipient.knowledge['guess'],
            "own_reasoning": recipient.knowledge['reasoning'],
            "receiver_name": recipient.name,
            "target_variable": context["target_variable"],
            "json_format_string": sender.knowledge_format}

def summary_game(sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = map_placeholders_summary_game(sender, recipient, context)
    prompt = generate_prompt_from_template(replacement_dict, prompt_template2)
    return {"role": role, "content": prompt}