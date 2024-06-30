import random
import numpy as np

# This class is essentially the model of the simulation.
# It includes the step functions for running the model
# as well as some key model properties. 
# TODO: I think I would prefer to structure this model
# in a similar way to the API for models in agents.jl
class DebateManager:
    def __init__(self, agents, correct_answer, num_rounds):
        self.agents = agents
        self.num_rounds = num_rounds
        self.correct_answer = int(correct_answer)  # Ensure correct_answer is an integer
        self.correct_counts_over_time = []  # Track correct answers over time
        self.tick = 0

    def run(self, parameters):
        print(f"Starting simulation with {len(self.agents)} agents.")
        
        # TODO: Move out of here. This is data that should be set much earlier.
        correct_agent = random.choice(self.agents)
        correct_agent.update_knowledge({"guess": self.correct_answer, "reasoning": "I have direct information that this is the correct answer."})

        for _ in range(self.num_rounds):
            self.step(parameters)

        return self.correct_counts_over_time
    
    def step(self, parameters):
        self.tick += 1
        print(f"Round {self.tick} begins.")
        shuffled_agents = random.sample(self.agents, len(self.agents))
        for i in range(0, len(shuffled_agents), 2):
            self.agent_step(shuffled_agents[i], parameters)
        correct_count = sum(agent.knowledge['guess'] == self.correct_answer for agent in self.agents)        
        self.correct_counts_over_time.append(correct_count)  # Add count of correct answers for the round
        print(f"Round {self.tick} ends. Correct answers: {correct_count}/{len(self.agents)}.")
        print("-" * 50)

    def agent_step(self, agent, parameters):
        graph = parameters["graph"]
        neighbour_ids = list(graph.neighbors(agent.name - 1))
        # Determine who the selected agent interacts with.
        neighbour_id = random.choice(neighbour_ids)
        self.exchange_information(agent, self.agents[neighbour_id], parameters)

    def exchange_information(self, agent1, agent2, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        prompt1 = construct_prompt_fn(agent1, agent2, {})
        prompt2 = construct_prompt_fn(agent2, agent1, {})
        chat_result_1 = agent1.initiate_chat(
            recipient=agent2, 
            message= prompt1,
            max_turns=1,
        )
        # TODO: We probably don't need two chats here since it is a two
        # way conversation.
        chat_result_2 = agent2.initiate_chat(
            recipient=agent1, 
            message= prompt2,
            max_turns=1,
        )
        # Split chat result between agent 1 and 2
        agent1.make_decision(chat_result_1.chat_history[-1]["content"])
        agent2.make_decision(chat_result_2.chat_history[-1]["content"])
    
