import random
import src.utils as utils
import src.data_utils as data_utils

class DebateManager:
    """This class manages the debate between agents.
    
    It is the model of the simulation.
    
    Model properties are initialized when instantiated and the
    `step` function specifies what happens when the model is run.
    
    See Agents.jl for a similar API for defining models."""
    
    def __init__(self, agents, params):
        names = ['correct_answer', 'num_rounds']
        correct_answer, num_rounds = [params[k] for k in names]
        self.simulation_id = params.get('simulation_id', 0)
        self.agents = agents
        self.num_rounds = num_rounds
        self.correct_answer = int(correct_answer)  # Ensure correct_answer is an integer
        self.initial_reasoning = params.get("initial_reasoning", "I have direct information that this is the correct answer.")
        self.tick = 0
        self.agent_results = []
        self.model_results = []

        correct_agent = random.choice(self.agents)
        self.source_node_id = correct_agent.agent_id
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
        correct_agent_ids = [agent.agent_id for agent in self.agents
                             if agent.knowledge['guess'] == self.correct_answer]
        misinformed_agent_ids = [agent.agent_id for agent in self.agents
                                 if agent.knowledge['guess'] != self.correct_answer]
        self.model_results.append({
                    'simulation_id': parameters.get('simulation_id', 0),
                    'round':  self.tick,
                    'source_node_id': self.source_node_id,
                    'correct_count': correct_count,
                    'correct_agent_ids': correct_agent_ids,
                    'correct_proportion': correct_count / len(self.agents),
                    'misinformed_agent_ids': misinformed_agent_ids,
                })
        print(f"Correct answers: {correct_count}/{len(self.agents)}.")
            
    def step(self, parameters):
        self.tick += 1
        shuffled_agents = random.sample(self.agents, len(self.agents))
        # Loop through agents in random order
        for i in range(len(shuffled_agents)):
            self.agent_step(shuffled_agents[i], parameters)
        
        if self.tick == parameters.get("info_shock_arrival_time"):
            self.apply_information_shock(parameters)

    def agent_step(self, agent, parameters):
        graph = parameters["graph"]
        neighbour_ids = list(graph.neighbors(agent.name - 1))
        # Determine who the selected agent interacts with.
        neighbour_id = random.choice(neighbour_ids)
        self.exchange_information(agent, self.agents[neighbour_id], parameters)

    def exchange_information(self, agent1, agent2, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        # Agent 1 receives a message from their peer agent 2 that exchanges info
        prompt = construct_prompt_fn(agent2, agent1, parameters)
        
        # Agent 1 chats to agent 2 to learn what agent 2 is thinking.
        # Only agent 1 updates their knowledge based on the conversation.
        # This ensures everyone updates their knowledge once per round.
        chat_result = agent2.initiate_chat(
            recipient=agent1, 
            message= prompt,
            max_turns=1,
        )
        
        # Extract data from the chat message and update the agent's knowledge.
        data_format = parameters.get("data_format", {"guess": int, "reasoning": str})
        message = chat_result.chat_history[-1]["content"]
        data = data_utils.extract_data(message, data_format)
        if len(data) >= 1:
            agent1.update_knowledge(data[0])
            print(f"Agent {agent1.name} updated knowledge to: {agent1.knowledge}")
        else:
            print(f"Agent {agent1.name} keeps its guess and reasoning unchanged: {agent1.knowledge}.")
            
    def apply_information_shock(self, parameters):
        """Apply a shock to the agents' knowledge."""
        shock_fn = parameters.get("info_shock_fn", None)
        if shock_fn is not None:
            shock_fn(self.agents, parameters)