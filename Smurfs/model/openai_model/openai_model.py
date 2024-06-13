from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Smurfs.model.base import BaseLM
import requests
import json

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
class OpenAI_Model(BaseLM):
    api_endpoint: str
    api_key: str
    url: str

    def __init__(self, *args, **kwargs):
        # self.messages = []
        api_endpoint = "https://api.ai-gaochao.cn/v1"
        api_key = "sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"
        url = f"{api_endpoint}/chat/completions"
        # self.model_name = model_name
        # self.method_name = method_name
        kwargs.update({"api_endpoint": api_endpoint})
        kwargs.update({"api_key": api_key})
        kwargs.update({"url": url})
        super().__init__(
            *args,
            **kwargs,
        )

    def prediction(self, messages, tools=None, tool_choice=None):
        """
        Get model response from a chat conversation, output the answer in assistant role.
        messages = [
            {"role": "system", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        """
        system = "You are AutoGPT, you can use many tools(functions) to do the following task."
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        message = [{'role': 'system', 'content': system}]
        message += messages

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "tools": tools,
            "tool_choice": tool_choice
        }
        
        # return prompt
        raw_response = requests.post(self.url, headers=headers, json=payload, verify=False)
        if tools == None:
            raw_response = json.loads(raw_response.content.decode("utf-8"))['choices'][0]['message']['content']
        else:
            raw_response = json.loads(raw_response.content.decode("utf-8"))['choices'][0]['message']

        return raw_response

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get the current weather",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "format": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"],
#                         "description": "The temperature unit to use. Infer this from the users location.",
#                     },
#                 },
#                 "required": ["location", "format"],
#             },
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_n_day_weather_forecast",
#             "description": "Get an N-day weather forecast",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "format": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"],
#                         "description": "The temperature unit to use. Infer this from the users location.",
#                     },
#                     "num_days": {
#                         "type": "integer",
#                         "description": "The number of days to forecast",
#                     }
#                 },
#                 "required": ["location", "format", "num_days"]
#             },
#         }
#     },
# ]

# question = "What's the weather like today. I'm in Glasgow, Scotland."
# worker = OpenAI_Model(model_name="gpt-4-0613")
# messages = [{"role": "user", "content": question}]
# responses = worker.prediction(messages)
# print(responses)