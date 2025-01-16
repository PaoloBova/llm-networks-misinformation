import logging
import random
import src.data_utils as data_utils
import src.networks as networks

class DebateManager:
    """This class manages the debate between agents.
    
    It is the model of the simulation.
    
    Model properties are initialized when instantiated and the
    `step` function specifies what happens when the model is run.
    
    See Agents.jl for a similar API for defining models."""
    
    def __init__(self, agents, params):
        names = ['correct_answer', 'num_rounds']
        correct_answer, num_rounds = [params[k] for k in names]
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
        self.graph = networks.init_graph(params)
    
    def collect_stats(self, parameters):
        for agent in self.agents:
            self.agent_results.append({
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
        graph = self.graph
        neighbour_ids = list(graph.neighbors(agent.agent_id - 1))
        # Determine who the selected agent interacts with.
        neighbour_id = random.choice(neighbour_ids)
        self.exchange_information(agent, self.agents[neighbour_id], parameters)

    def exchange_information(self, agent1, agent2, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        # Agent 1 receives a message from their peer agent 2 that exchanges info
        args = {**parameters, "tick": self.tick}
        prompt = construct_prompt_fn(agent2, agent1, args)
        
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
            

class TechnologyLearningGame:
    """This class manages the technology learning game played on a network.
    
    It is the model of the simulation.
    
    Model properties are initialized when instantiated and the
    `step` function specifies what happens when the model is run.
    
    See Agents.jl for a similar API for defining models."""
    
    def __init__(self, agents, params):
        names = ['num_rounds']
        num_rounds, = [params[k] for k in names]
        self.agents = agents
        self.num_rounds = num_rounds
        self.tick = 0
        self.agent_results = []
        self.model_results = []
        # In this model, we need a dummy adjudicator agent to prompt the agents.
        # This should be a ConversableAgent from Autogen but the agent does not
        # need a system prompt because we only use the agent to prompt the other
        # agents in the model using the initiate_chat method. We do it this way
        # to be more consisent with similar models where an adjudicator agent is
        # necessary for evaluating the agents' responses.
        self.adjudicator_agent = params.get("adjudicator_agent")
        params["num_agents"] = len(agents)
        self.graph = networks.init_graph(params)
        
        if params.get('hq_chance', 1) > 0.5:
            self.correct_answer = 1
        else:
            self.correct_answer = 0
    
    def collect_stats(self, parameters):
        agent_result_prev = self.agent_results[-len(self.agents):]
        for agent in self.agents:
            self.agent_results.append({
                    'round': self.tick,
                    'agent_id': agent.name,
                    'decision': agent.knowledge.get('decision'),
                    'reasoning': agent.knowledge.get('reasoning')
                })

        correct_count = sum(agent.knowledge['decision'] == self.correct_answer
                            for agent in self.agents)
        # Score how correct/incorrect the consenus among the agents is
        n0 = parameters["initial_share_correct"]
        n = parameters["num_agents"]
        nt = correct_count
        assert n > 0
        if n == n0:
            ct = nt / n
            logging.warning("All agents have the same initial decision.")
        elif n0 == 0:
            ct = nt / n
            logging.warning("No agents have the correct initial decision.")
        else:
            ct = (nt - n0) / (n - n0) if nt >= n0 else (nt - n0) / n0
        # Record the proportion of agents who switched decisions
        decisions_old = [r['decision'] for r in agent_result_prev]
        decisions_new = [agent.knowledge['decision'] for agent in self.agents]
        switches =  [i != j for i, j in zip(decisions_old, decisions_new)]
        if self.tick == 0:
            switch_rate = None
        else:
            switch_rate = sum(switches) / len(self.agents)
        correct_agent_ids = [agent.agent_id for agent in self.agents
                             if agent.knowledge['decision'] == self.correct_answer]
        misinformed_agent_ids = [agent.agent_id for agent in self.agents
                                 if agent.knowledge['decision'] != self.correct_answer]
        self.model_results.append({
                    'round':  self.tick,
                    'correct_count': correct_count,
                    'correct_agent_ids': correct_agent_ids,
                    'correct_proportion': correct_count / len(self.agents),
                    'consensus_score': ct,
                    'switch_rate': switch_rate,
                    'misinformed_agent_ids': misinformed_agent_ids,
                })
        logging.info(f"Correct answers: {correct_count}/{len(self.agents)}.")
            
    def step(self, parameters):
        self.tick += 1
        shuffled_agents = random.sample(self.agents, len(self.agents))
        # Loop through agents in random order
        for i in range(len(shuffled_agents)):
            self.agent_step(shuffled_agents[i], parameters)

    def agent_step(self, agent, parameters):
        self.respond_to_system(agent, parameters)
        new_decision = agent.knowledge.get('decision')
        agent.state["decision_old"] = agent.state.get("decision")
        agent.state["decision"] = new_decision
        
        # In a variant of this game, we don't compute the utilities
        # until the end, so here we skip this step.
        
        if not parameters.get("compute_utilities_at_end", False):
            self.compute_utilities(agent, parameters)
    
    def compute_utilities(self, agent, parameters):
        # Agent gains utility based on their decision
        # It is well known that A gives you 1 utility with 0.5 chance, and 0 otherwise.
        # Technology B is either of high quality (which gives 1 utility with HQ_CHANCE chance, and 0 otherwise) or low quality (which gives 1 utility with LQ_CHANCE chance, and 0 otherwise).
        hq_chance = parameters.get("hq_chance", 1)
        lq_chance = 1 - hq_chance
        if agent.state["decision"] == 1:
            true_quality = parameters.get("true_quality", 0)
            chance = hq_chance if true_quality == 1 else lq_chance
            new_utility = random.choices([1, 0], weights=[chance, 1- chance])[0]
        else:
            new_utility = random.choices([1, 0], weights=[0.5, 0.5])[0]
        agent.state["utility_gained"] = new_utility
        if "utility" not in agent.state:
            agent.state["utility"] = 0
        agent.state["utility"] += new_utility
    
    def respond_to_system(self, agent, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        # The agent is prompted to reflect on the current state of the simulation
        # and update their knowledge or beahviour accordingly.
        args = {**parameters,
                "tick": self.tick,
                "agents": self.agents,
                "graph": self.graph}
        adjudicator = self.adjudicator_agent
        prompt = construct_prompt_fn(adjudicator, agent, args)
        logging.info(f"Prompting agent {agent.name} with: {prompt}")
        chat_result = adjudicator.initiate_chat(
            recipient=agent, 
            message= prompt,
            max_turns=1,
            clear_history=False,
            silent=True,
        )
        
        # Extract data from the chat message and update the agent's knowledge.
        data_format = agent.knowledge_format if hasattr(agent, "knowledge_format") else {}
        message = chat_result.chat_history[-1]["content"]
        logging.info(f"Agent {agent.name} received message: {message}")
        data = data_utils.extract_data(message, data_format)
        if len(data) >= 1:
            agent.update_knowledge(data[0])
