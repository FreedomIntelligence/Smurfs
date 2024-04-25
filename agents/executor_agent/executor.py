from Multi_agents.agents.base import BaseAgent
from Multi_agents.agents.executor_agent.prompt import generate_thought_prompt, choose_tool_prompt, choose_parameter_prompt

class executor_agent(BaseAgent):
    def __init__(self, llm, logger_dir):
        self.thought_prompt = generate_thought_prompt
        self.tool_prompt = choose_tool_prompt
        self.parameter_prompt = choose_parameter_prompt
        self.llm = llm
        self.name = "Executor Agent"
        self.logger_dir = logger_dir

    def run(self, query_id, task, **kwargs):
        """agent run one step"""
        if task == "thought":
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            ind = 0

            while True:
                try:
                    result = self.llm.prediction(message)
                    self.log(query_id, result)
                    print(result)
                    return result
                except Exception as e:
                    print(f"generating thought fails: {e}")
                    self.log(query_id, f"generating thought fails: {e}")
                    if ind > self.max_retry:
                        return -1
                    ind += 1
                    continue
        
        elif task == "tool":
            thought = kwargs["thought"]
            kwargs["question"] = kwargs["question"]+f"thought: {thought}\n"
            del kwargs["thought"]
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message)
                    start = result.find("{")
                    end = result.rfind("}")
                    result = eval(result[start:end+1])
                    print(result)
                    tool = result['ID']
                    self.log(query_id, result)
                    return tool
                except Exception as e:
                    print(f"choosing tool fails: {e}")
                    self.log(query_id, f"choosing tool fails: {e}")
                    if ind > self.max_retry:
                        return -1
                    ind += 1
                    continue

        elif task == "parameter":
            thought = kwargs["thought"]
            del kwargs["thought"]
            kwargs["question"] = kwargs["question"]+f"thought: {thought}\n"
            api_dic = kwargs["api_dic"]
            if len(api_dic["required_parameters"]) == 0 and len(api_dic["optional_parameters"]) == 0:
                return {}
            self.get_memory(**kwargs)
            agent_prompt = self.get_prompt(task)
            message = [{'role': 'user', 
                    'content': agent_prompt}]
            
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message)
                    start = result.find("{")
                    end = result.rfind("}")
                    print(result[start:end+1])
                    result = result[start:end+1]
                    clean_answer = eval(
                        result.replace(": true", ": True").replace(": false", ": False").replace("```", "").strip())
                    # a = clean_answer["Parameters"]
                    # clean_answer = clean_answer["Parameters"]
                    self.log(query_id, clean_answer)
                    return clean_answer
                except Exception as e:
                    print(f"choose parameter fails: {e}")
                    self.log(query_id, f"choose parameter fails: {e}")
                    if ind > self.max_retry:
                        return -1
                    ind += 1
                    continue

    def get_memory(self, **kwargs):
        """get relevant memory and add it to agent's memory"""
        self.memory = kwargs
    
    def get_prompt(self, task):
        """get the prompt for the agent"""
        if task == "thought":
            agent_prompt = self.thought_prompt.format(**self.memory)
        elif task == "tool":
            agent_prompt = self.tool_prompt.format(**self.memory)
        elif task == "parameter":
            agent_prompt = self.parameter_prompt.format(**self.memory)
            
        return agent_prompt