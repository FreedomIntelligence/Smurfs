from abc import abstractmethod
from pydantic import BaseModel, Field
from model.base import BaseLM
# import json
import os
from langchain.prompts import PromptTemplate

class BaseAgent(BaseModel):
    name: str
    llm: BaseLM
    memory: dict
    prompt: PromptTemplate
    logger_dir: str
    max_retry: int = Field(default=10)

    @abstractmethod
    def run(self, **kwargs):
        """agent run one step"""
        pass

    @abstractmethod
    def get_memory(self, **kwargs):
        """get relevant memory and add it to agent's memory"""
        pass
    
    @abstractmethod
    def get_prompt(self, **kwargs):
        """get the prompt for the agent"""
        pass
    
    def log(self, query_id, content):
        """write log to the logger file"""
        logger_file = os.path.join(self.logger_dir, f"{query_id}.txt")
        with open(logger_file, "a+") as file:
            file.write("\n##########\n")
            file.write(f"{self.name}: \n\n")
            file.write(content)
            file.write("\n##########\n")