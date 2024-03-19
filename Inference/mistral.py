from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

class Mistral_Model():
    def __init__(self, model_path, device="cuda"):
        # self.messages = []
        self.device = device
        self.gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':4096, 'num_beams':1,
                'do_sample':True, 'top_p':0.95, 'temperature':0.95, 'repetition_penalty':1.0}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).half()
        except Exception as e:
            print(e)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True).half()
        if self.tokenizer.pad_token_id == None:
            # print("No pad_token_id in tokenizer")
            self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def get_response(self,inputs,outputs,tokenizer,num_return):
        responses_list=[]
        batch_return=[]
        for i, output in enumerate(outputs):
            input_len = len(inputs[0])
            generated_output = output[input_len:]
            batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
            if i%num_return==num_return-1:
                responses_list.append(batch_return)
                batch_return=[]
        return responses_list

    def get_prompt(self, messages):
        prompt  = ""
        for m in messages:
            if m["role"] == "system":
                content = m["content"]
                prompt += f"<s>[INST] {content} [/INST]"
            if m["role"] == "user":
                content = m["content"]
                prompt += f"[INST] {content} [/INST]"
            if m["role"] == "assistant":
                content = m["content"]
                prompt += f" {content}</s>"
        return prompt

    def prediction(self, messages):
        """
        Get model response from a chat conversation, output the answer in assistant role.
        messages = [
            {"role": "system", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ]
        """
        # if self.messages == []:
        #     self.messages = messages
        logger_path = "/home/chenjunzhi/Modulized_Prompt_LLM/data/log.txt"
        log_file = open(logger_path, "a+")
        accelerator = Accelerator()
        prompts = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        prompt = self.tokenizer.decode(prompts[0])
        log_file.write(prompt)
        # print(prompt)
        # print("\n\n")
        input_ids = prompts
        # print("####1####")
        model,input_ids = accelerator.prepare(self.model,input_ids)
        input_ids = input_ids.to('cuda')
        outputs = accelerator.unwrap_model(model).generate(input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_kwargs)
        response = self.get_response(input_ids, outputs, self.tokenizer, 1)
        # print("####2####")
        # assistant = {"role": "assistant", "content": response[0][0]}
        # self.messages.append(assistant)
        log_file.write(response[0][0])
        log_file.write("\n\n\n\n")
        log_file.close()
        return response[0][0]
        # tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # out = self.model.generate(tokenized_chat, **self.gen_kwargs)
        # return self.tokenizer.decode(out[0])
    
    # def clear_chat_history(self):
    #     self.messages = []

# mis = Mistral_Model("/mntcephfs/lab_data/chenjunzhi/mistralai:Mistral-7B-Instruct-v0.2")
# messages = [
#             {"role": "user", "content": "What is your favourite condiment?"},
#             {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#             {"role": "user", "content": "Do you have mayonnaise recipes?"}
#         ]
# response = mis.prediction(messages)
# print(response)