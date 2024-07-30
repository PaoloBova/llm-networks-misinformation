import pytest
from src.agent import Agent
from src.prompts import prompt_fn_example, baseline_game
import autogen
from typing import Dict

# Define your test functions with the prefix 'test_'

def test_prompt_fn_example():
    # Arrange
    knowledge1 = {"guess": "42", "reasoning": "I have direct information that this is the correct answer."}
    knowledge2 = {"guess": "42", "reasoning": "I have direct information that this is the correct answer."}
    sender = Agent(1, "apikey", 0, model="gpt-3.5-turbo", knowledge=knowledge1)
    recipient = Agent(2, "apikey", 0, model="gpt-3.5-turbo", knowledge=knowledge2)
    prompt_expected1 = """You've received information that the number might be 42 because 'I have direct information that this is the correct answer.'. Recall that you currently believe 42 because of the following reason: "I have direct information that this is the correct answer.". Consider whether you should update your beliefs. Give your new guess and reasoning in json format even if your answer is unchanged: {"guess": int, "reasoning": str}"""

    # Act
    prompt_result1 = prompt_fn_example(sender, recipient, {})['content']

    # Assert
    assert prompt_result1 == prompt_expected1, f"Expected: {prompt_expected1}, but got: {prompt_result1}"

def test_prompt_fn1():
    # Arrange
    knowledge1 = {"guess": "42", "reasoning": "I have direct information that this is the correct answer."}
    knowledge2 = {"guess": "42", "reasoning": "I have direct information that this is the correct answer."}
    sender = Agent(1, "apikey", 0, model="gpt-3.5-turbo", knowledge=knowledge1)
    recipient = Agent(2, "apikey", 0, model="gpt-3.5-turbo", knowledge=knowledge2)
    prompt_expected1 = """You've received information that the number might be 42 because 'I have direct information that this is the correct answer.'. Recall that you currently believe 42 because of the following reason: "I have direct information that this is the correct answer.". Consider whether you should update your beliefs. Give your new guess and reasoning in json format even if your answer is unchanged: {"guess": int, "reasoning": str}"""

    # Act
    prompt_result2 = baseline_game(sender, recipient, {})['content']

    # Assert
    assert prompt_result2 == prompt_expected1, f"Expected: \n{prompt_expected1}, \n but got: \n{prompt_result2}"