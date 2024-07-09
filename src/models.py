import random

class DebateManager:
    """This class manages the debate between agents.
    
    It is the model of the simulation.
    
    Model properties are initialized when instantiated and the
    `step` function specifies what happens when the model is run.
    
    See Agents.jl for a similar API for defining models."""
    
    def __init__(self, agents, params):
        names = ['correct_answer', 'num_rounds', 'seed']
        correct_answer, num_rounds, seed = [params[k] for k in names]
        self.agents = agents
        self.num_rounds = num_rounds
        self.correct_answer = int(correct_answer)  # Ensure correct_answer is an integer
        self.initial_reasoning = params.get("initial_reasoning", "I have direct information that this is the correct answer.")
        self.tick = 0
        self.agent_results = []
        self.model_results = []

        correct_agent = random.choice(self.agents)
        correct_agent.update_knowledge({"guess": self.correct_answer,
                                        "reasoning": self.initial_reasoning})
    
    def collect_stats(self, parameters):
        for agent in self.agents:
            self.agent_results.append({
                    'simulation_id': parameters.get('simulation_id', 0),  # Default to 0 if 'simulation_id' is not in parameters
                    'round': self.tick,
                    'agent_id': agent.name,
                    'guess': agent.knowledge['guess'],
                    'reasoning': agent.knowledge['reasoning']
                })

            correct_count = sum(agent.knowledge['guess'] == self.correct_answer
                                for agent in self.agents)
            self.model_results.append({
                        'simulation_id': parameters.get('simulation_id', 0),
                        'round':  self.tick,
                        'correct_count': correct_count,
                        'correct_proportion': correct_count / len(self.agents)
                    })
            print(f"Correct answers: {correct_count}/{len(self.agents)}.")
            
    def step(self, parameters):
        self.tick += 1
        shuffled_agents = random.sample(self.agents, len(self.agents))
        for i in range(0, len(shuffled_agents), 2):
            self.agent_step(shuffled_agents[i], parameters)

    def agent_step(self, agent, parameters):
        graph = parameters["graph"]
        neighbour_ids = list(graph.neighbors(agent.name - 1))
        # Determine who the selected agent interacts with.
        neighbour_id = random.choice(neighbour_ids)
        self.exchange_information(agent, self.agents[neighbour_id], parameters)

    def exchange_information(self, agent1, agent2, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        prompt1 = construct_prompt_fn(agent1, agent2, parameters)
        prompt2 = construct_prompt_fn(agent2, agent1, parameters)
        
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
    
