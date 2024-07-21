from abc import abstractmethod
from pydantic import BaseModel, Field

class BaseLM(BaseModel):
    model_name: str
    sys_prompt: str = "You are AutoGPT, you can use many tools(functions) to do the following task."
    
    @abstractmethod
    def prediction(self, *args, **kwargs):
        """Use the llm to generate the sequence"""
        pass

    def change_sys_prompt(self, content):
        self.sys_prompt = content
    
    def set_default_sys_prompt(self):
        self.sys_prompt = "You are AutoGPT, you can use many tools(functions) to do the following task."