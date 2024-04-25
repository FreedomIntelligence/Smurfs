from Multi_agents.agents.base import BaseAgent
from Multi_agents.agents.answer_agent.prompt import answer_generation_direct_prompt, answer_generation_prompt, final_answer_generation_prompt, tool_check_prompt

class answer_agent(BaseAgent):
    def __init__(self, llm, logger_dir):
        self.direct_prompt = answer_generation_direct_prompt
        self.answer_prompt = answer_generation_prompt
        self.final_prompt = final_answer_generation_prompt
        self.tool_check_prompt = tool_check_prompt
        self.llm = llm
        self.name = "Answer Agent"
        self.logger_dir = logger_dir

    def run(self, query_id, task, **kwargs):
        """agent run one step"""
        if task == "direct":
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            result = self.llm.prediction(message)
            self.log(query_id, result)
            print(result)
            return result
        
        elif task == "answer":
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message)
                    self.log(query_id, result)
                    break
                except Exception as e:
                    print(f"answer generation fails: {e}")
                    self.log(query_id, f"answer generation fails: {e}")
                    if ind > 2:
                        return -1
                    ind += 1
                    continue
            return result

        elif task == "final":
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message)
                    self.log(query_id, result)
                    break
                except Exception as e:
                    print(f"answer generation fails: {e}")
                    self.log(query_id, f"answer generation fails: {e}")
                    if ind > 2:
                        return -1
                    ind += 1
                    continue
            return result
        
        elif task == "tool_check":
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message)
                    result = eval(result)
                    a = result["Reason"]
                    b = result["Choice"]
                    self.log(query_id, result)
                    if 'yes' in b.lower():
                        return result, -1
                    else:
                        return result, 1
                except Exception as e:
                    print(f"tool check fails: {e}")
                    self.log(query_id, f"tool check fails: {e}")
                    if ind > self.max_retry:
                        return "", -1
                    ind += 1
                    continue

    def get_memory(self, **kwargs):
        """get relevant memory and add it to agent's memory"""
        self.memory = kwargs
    
    def get_prompt(self, task):
        """get the prompt for the agent"""
        if task == "direct":
            agent_prompt = self.direct_prompt.format(**self.memory)
        elif task == "answer":
            agent_prompt = self.answer_prompt.format(**self.memory)
        elif task == "final":
            agent_prompt = self.final_prompt.format(**self.memory)
        elif task == "tool_check":
            agent_prompt = self.tool_check_prompt.format(**self.memory)
            
        return agent_prompt