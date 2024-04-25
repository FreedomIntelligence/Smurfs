from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from Multi_agents.model.base import BaseModel

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
class vllm_Model(BaseModel):
    def __init__(self, model_name):
        # self.messages = []
        openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "EMPTY"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name

    def prediction(self, messages):
        """
        Get model response from a chat conversation, output the answer in assistant role.
        messages = [
            {"role": "system", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.95
        )
        raw_response = response.choices[0].message.content
        return raw_response