from Smurfs.agents.base import BaseAgent
from Smurfs.agents.executor_agent.prompt import generate_thought_prompt, choose_tool_prompt, choose_parameter_prompt
from Smurfs.inference.utils import change_name, standardize, contain
from Smurfs.inference.server import get_rapidapi_response
from typing import Any
import json
import os
import time
import requests

class executor_agent(BaseAgent):
    thought_prompt: Any
    tool_prompt: Any
    parameter_prompt: Any

    def __init__(self, *args, **kwargs):
        thought_prompt = generate_thought_prompt
        tool_prompt = choose_tool_prompt
        parameter_prompt = choose_parameter_prompt
        name = "Executor Agent"
        kwargs.update({"thought_prompt": thought_prompt})
        kwargs.update({"tool_prompt": tool_prompt})
        kwargs.update({"parameter_prompt": parameter_prompt})
        kwargs.update({"name": name})
        super().__init__(
            *args,
            **kwargs,
        )

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
                    self.colorful_print(result, "Thought Generation")
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
                    self.colorful_print(result, "Choose Tool")
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
                    self.colorful_print(result[start:end+1], "Generate Parameters")
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

class stream_executor_agent(BaseAgent):
    thought_prompt: Any
    tool_prompt: Any
    parameter_prompt: Any

    def __init__(self, *args, **kwargs):
        thought_prompt = generate_thought_prompt
        tool_prompt = choose_tool_prompt
        parameter_prompt = choose_parameter_prompt
        name = "Executor Agent"
        kwargs.update({"thought_prompt": thought_prompt})
        kwargs.update({"tool_prompt": tool_prompt})
        kwargs.update({"parameter_prompt": parameter_prompt})
        kwargs.update({"name": name})
        super().__init__(
            *args,
            **kwargs,
        )

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
                    self.colorful_print(result, "Thought Generation")
                    return result, "Thought Generation", self.name, result
                except Exception as e:
                    print(f"generating thought fails: {e}")
                    self.log(query_id, f"generating thought fails: {e}")
                    if ind > self.max_retry:
                        return -1, "Thought Generation", self.name, str(e)
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
                    self.colorful_print(result, "Choose Tool")
                    tool = result['ID']
                    self.log(query_id, result)
                    return tool, "Choose Tool", self.name, result
                except Exception as e:
                    print(f"choosing tool fails: {e}")
                    self.log(query_id, f"choosing tool fails: {e}")
                    if ind > self.max_retry:
                        return -1, "Choose Tool", self.name, str(e)
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
                    self.colorful_print(result[start:end+1], "Generate Parameters")
                    result = result[start:end+1]
                    clean_answer = eval(
                        result.replace(": true", ": True").replace(": false", ": False").replace("```", "").strip())
                    # a = clean_answer["Parameters"]
                    # clean_answer = clean_answer["Parameters"]
                    self.log(query_id, clean_answer)
                    return clean_answer, "Generate Parameters", self.name, result[start:end+1]
                except Exception as e:
                    print(f"choose parameter fails: {e}")
                    self.log(query_id, f"choose parameter fails: {e}")
                    if ind > self.max_retry:
                        return -1, "Generate Parameters", self.name, str(e)
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

