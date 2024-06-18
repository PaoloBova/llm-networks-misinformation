import random
from typing import Union, Dict
import numpy as np
from autogen import ConversableAgent

class DebateManager:
    def __init__(self, agents, correct_answer, num_rounds):
        self.agents = agents
        self.num_rounds = num_rounds
        self.correct_answer = int(correct_answer)  # Ensure correct_answer is an integer
        self.correct_counts_over_time = []  # Track correct answers over time
        self.tick = 0

    def start_debate_rounds(self):
        print(f"Starting simulation with {len(self.agents)} agents.")
        correct_agent = random.choice(self.agents)
        correct_agent.update_knowledge({"guess": self.correct_answer, "reasoning": "I have direct information that this is the correct answer."})

        for _ in range(self.num_rounds):
            self.step()

        return self.correct_counts_over_time
    
    def step(self):
        self.tick += 1
        print(f"Round {self.tick} begins.")
        self.conduct_round()
        correct_count = sum(agent.knowledge['guess'] == self.correct_answer for agent in self.agents)        
        self.correct_counts_over_time.append(correct_count)  # Add count of correct answers for the round
        print(f"Round {self.tick} ends. Correct answers: {correct_count}/{len(self.agents)}.")
        print("-" * 50)

    def conduct_round(self):
        shuffled_agents = random.sample(self.agents, len(self.agents))
        for i in range(0, len(shuffled_agents), 2):
            if i + 1 < len(shuffled_agents):
                agent1, agent2 = shuffled_agents[i], shuffled_agents[i + 1]
                self.exchange_information(agent1, agent2)

    # TODO: Careful: max_turns parameter is not working.
    def exchange_information(self, agent1, agent2):
        prompt1 = self.construct_prompt(agent1, agent2, {})
        prompt2 = self.construct_prompt(agent2, agent1, {})
        chat_result_1 = agent1.initiate_chat(
            recipient=agent2, 
            message= prompt1,
            max_turns=1,
        )
        chat_result_2 = agent2.initiate_chat(
            recipient=agent1, 
            message= prompt2,
            max_turns=1,
        )
        # Split chat result between agent 1 and 2
        agent1.make_decision(chat_result_1.chat_history[-1]["content"])
        agent2.make_decision(chat_result_2.chat_history[-1]["content"])
    
    def construct_prompt(self, sender: ConversableAgent, recipient: ConversableAgent, context: dict) -> Union[str, Dict]:
        guess = sender.knowledge['guess']
        reasoning = sender.knowledge['reasoning']
        own_guess = recipient.knowledge['guess']
        own_reasoning = recipient.knowledge['reasoning']
        json_format_string = """{"label": str, "explanation": str}"""
        return {
            "role": "system",
            "content": f"You've received information that the number might be {guess} because '{reasoning}'. Recall that you currently believe {own_guess} because {own_reasoning}. Consider whether you should update your beliefs. Give your new guess and reasoning in json format even if your answer is unchanged: {json_format_string}"
        }