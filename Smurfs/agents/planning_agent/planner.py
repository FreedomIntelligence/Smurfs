from Smurfs.agents.base import BaseAgent
from Smurfs.agents.planning_agent.prompt import task_decompose_prompt

class planning_agent(BaseAgent):
    def __init__(self, llm, logger_dir):
        super().__init__(
            prompt = task_decompose_prompt,
            llm = llm,
            name = "Planning Agent",
            logger_dir = logger_dir
        )

    def run(self, question, query_id):
        """agent run one step"""
        self.get_memory(question)
        agent_prompt = self.get_prompt()
        message = [{'role': 'user', 
                 'content': agent_prompt}]
        ind = 0
        while True:
            try:
                result = self.llm.prediction(message)
                # print(result)
                start = result.find("{")
                end = result.find("}")
                result = eval(result[start:end+1])
                print(result)
                subtasks = result['Tasks']
                self.log(query_id, result)
                # print(a)
                return subtasks
            except Exception as e:
                print(f"task deompose fails: {e}")
                self.log(query_id, f"task deompose fails: {e}")
                if ind > self.max_retry:
                    return -1
                ind += 1
                continue

    def get_memory(self, question):
        """get relevant memory and add it to agent's memory"""
        self.memory = {"question": question}
    
    def get_prompt(self):
        """get the prompt for the agent"""
        agent_prompt = self.prompt.format(**self.memory)
        return agent_prompt