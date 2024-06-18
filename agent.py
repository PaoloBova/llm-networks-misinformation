import requests
import json
import random
import re
import autogen

class Agent(autogen.ConversableAgent):
    def __init__(self, agent_id, api_key, temperature, model="gpt-3.5-turbo", knowledge=None):
        super().__init__(name=agent_id,
                         llm_config={"model": model,
                                     "temperature": temperature})
        self.knowledge = knowledge or {"guess": random.randint(1, 100),
                                       "reasoning": "Initial random guess."}

    def update_knowledge(self, new_knowledge):
        self.knowledge = new_knowledge

    def make_decision(self, new_message_content):
        print("Making decision \n")

        if "direct information" in new_message_content or "reliable source" in new_message_content:
            guess_match = re.search(r'\b\d+\b', new_message_content)
            if guess_match:
                new_guess = int(guess_match.group())
                self.knowledge['guess'] = new_guess
                self.knowledge['reasoning'] = "I heard from a reliable source."
                print(f"Agent {self.name} updated knowledge to: {self.knowledge}")
        else:
            # Keep the existing guess and reasoning if the new information isn't considered more credible
            print(f"Agent {self.name} keeps its guess and reasoning unchanged: {self.knowledge}.")
            
        return len(new_message_content.split())
