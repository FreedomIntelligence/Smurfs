from abc import abstractmethod
from pydantic import BaseModel, Field

class BaseLM(BaseModel):
    model_name: str
    
    @abstractmethod
    def prediction(self, *args, **kwargs):
        """Use the llm to generate the sequence"""
        pass