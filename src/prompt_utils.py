import os
from typing import Dict
import autogen

def generate_prompt_from_template_path(replacement_dict: dict,
                                       template_path: str) -> str:
    """
    Generate a prompt based on the provided replacement data and a template.
    
    Parameters:
    - replacement_dict: A dictionary where keys are placeholders in the template and values are the replacements.
    - template_path: Path to the template file.
    
    Returns:
    - A string containing the updated prompt.
    """
    # Load the template
    with open(template_path, 'r') as file:
        template = file.read()
    
    # Replace placeholders with provided data
    for placeholder, replacement in replacement_dict.items():
        template = template.replace(f"[{placeholder}]", str(replacement))
    
    return template


def generate_prompt_from_template(replacement_dict: dict,
                                  template: str) -> str:
    """
    Generate a prompt based on the provided replacement data and a template.
    
    Parameters:
    - replacement_dict: A dictionary where keys are placeholders in the template and values are the replacements.
    - template: template with placeholder text provided as a string.
    
    Returns:
    - A string containing the updated prompt.
    """
    
    # Replace placeholders with provided data
    for placeholder, replacement in replacement_dict.items():
        template = template.replace(f"{{{placeholder}}}", str(replacement))
    
    return template
