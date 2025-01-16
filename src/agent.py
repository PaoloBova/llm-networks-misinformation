import autogen
import random

class Agent(autogen.ConversableAgent):
    def __init__(self, agent_id, api_key, temperature, model="gpt-3.5-turbo", knowledge=None):
        super().__init__(name=str(agent_id),
                         llm_config={"model": model,
                                     "api_key": api_key,
                                     "temperature": temperature})
        self.knowledge = knowledge or {"guess": random.randint(1, 100),
                                       "reasoning": "Initial random guess."}
        self.knowledge_format = """{"guess": int, "reasoning": str}"""
        self.agent_id = agent_id

    def update_knowledge(self, new_knowledge):
        self.knowledge = new_knowledge

class NetworkAgent(autogen.ConversableAgent):
    def __init__(self,
                 agent_id,
                 api_key,
                 temperature,
                 model="gpt-3.5-turbo",
                 knowledge=None,
                 knowledge_format={"reasoning": str, "decision": int},
                 state={}):
        super().__init__(name=str(agent_id),
                         human_input_mode="NEVER",
                         llm_config={"model": model,
                                     "api_key": api_key,
                                     "response_format": {"type": "json_object"},
                                     "temperature": temperature})
        self.knowledge = knowledge or {"guess": random.randint(1, 100),
                                       "reasoning": "Initial random guess."}
        self.knowledge_format = knowledge_format
        self.agent_id = agent_id
        self.state = state

    def update_knowledge(self, new_knowledge):
        self.knowledge = new_knowledge