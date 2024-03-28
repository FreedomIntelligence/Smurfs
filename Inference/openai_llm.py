from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import json

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
class OpenAI_Model():
    def __init__(self, model_name, method_name):
        # self.messages = []
        self.api_endpoint = "https://api.ai-gaochao.cn/v1"
        self.api_key = "sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"
        self.url = f"{self.api_endpoint}/chat/completions"
        self.model_name = model_name
        self.method_name = method_name

    def prediction(self, messages):
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
            "temperature": 0.95
        }
        
        # return prompt
        raw_response = requests.post(self.url, headers=headers, json=payload, verify=False)
        raw_response = json.loads(raw_response.content.decode("utf-8"))['choices'][0]['message']['content']
        method_name = self.method_name
        logger_path = f"data/{method_name}_log.txt"
        log_file = open(logger_path, "a+")
        log_file.write(str(messages))
        log_file.write("/n/n")
        log_file.write(str(raw_response))
        log_file.write("\n\n\n\n")
        log_file.close()

        return raw_response
