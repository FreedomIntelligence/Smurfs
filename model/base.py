from abc import abstractmethod
from pydantic import BaseModel, Field

class BaseLM(BaseModel):
    model_name: str
    
    @abstractmethod
    def prediction(self, messages):
        """Use the llm to generate the sequence"""
        pass