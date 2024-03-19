from bm25 import BM25
import json
import random

#每一段对话都有一个id, id要改成index数，可以直接通过id访问该对话
#每一个role加一个id field，这样就可以通过两个id找到对话
#需要用bm25搜索：query，thought
#直接匹配：action，存一个字典，每个action一个key对应一个列表，存储二元组（对话id， role id）
#要看（query，thought）pair 和 （thought，action）pair 还有 （assistant，assistant+1[thought]）pair
#先定位到最相近的对话index再做拆分把


diversity_prompt = "This is not the first time you try this task, all previous trails failed."
DEFAULT_THOUGHT_TOKEN = "Thought: "
DEFAULT_ACTION_TOKEN = "Action: "
DEFAULT_INPUTS_TOKEN = "Action Input: "
DEFAULT_OBSERVATION_TOKEN = "Observation: "

class tool_database:
    def __init__(self, base_path, tokenizer, tool_doc_path, retrieve_method="bm25"):
        self.tokenizer = tokenizer
        self.retrieve_method = retrieve_method
        with open(base_path, 'r') as file:
            self.data = json.load(file)
        with open(tool_doc_path, 'r') as file:
            self.tool_doc = json.load(file)
        #action: (i, j)
        self.action_database = {}
        #(i, j, query)
        self.query_database = []
        #(i, j, thought)
        self.thought_database = []
        #action: action_des(dic)
        self.tool_doc_database = {}
        #tool_name: tool_des
        self.tool_des = {}
        self.initialize_database()
        self.bm25_query = BM25(self.query_database, tokenizer)
        self.bm25_thought = BM25(self.thought_database, tokenizer)

    
    def initialize_database(self):
        for i in range(len(self.data)):
            self.data[i]["id"] = i
            for j in range(len(self.data[i]["conversations"])):
                # self.data[i]["conversations"][j]["id"] = j
                current_conv = self.data[i]["conversations"][j]

                if current_conv["from"] == "user":
                    if not current_conv["value"].startswith(diversity_prompt):
                        self.query_database.append((i, j, current_conv["value"]))
                
                if current_conv["from"] == "assistant":
                    assistant = self.assistant_decompose(current_conv["value"])
                    if isinstance(assistant, dict):
                        if not isinstance(assistant["thought"], str):
                            continue
                        self.thought_database.append((i, j, assistant["thought"]))
                        if assistant["action"] not in self.action_database.keys():
                            self.action_database[assistant["action"]] = []
                        self.action_database[assistant["action"]].append((i, j))
        
        for t in self.tool_doc.values():
            tool_name = t["tool_name"]
            tool_description = t["tool_description"]
            self.tool_des[tool_name] = f"{tool_name}: {tool_description}"
            for tool in t["tool_guidelines"].keys():
                new_key = f"{tool_name}.{tool}"
                self.tool_doc_database[new_key] = t["tool_guidelines"][tool]


    def parse_output(self, output: str): 

        def _extract(text, start_string, end_string):
            if text is None or start_string is None:
                return None
            
            begin = text.rfind(start_string)
            if end_string is not None:
                end = text.rfind(end_string)
            else:
                end = len(text)

            if begin == -1 or end == -1 or begin > end:
                return None
            content = text[begin+len(start_string): end].strip()
            if content == None:
                return None
            return content
    # <|thought|> 我可以直接回答这个问题。<|action|> null <|input|> null <|observation|> null <|response|> 1+1=2
        thought = _extract(output, DEFAULT_THOUGHT_TOKEN, DEFAULT_ACTION_TOKEN)
        action = _extract(output, DEFAULT_ACTION_TOKEN, DEFAULT_INPUTS_TOKEN)
        inputs = _extract(output, DEFAULT_INPUTS_TOKEN, None)
    
        return {
            "thought": thought,
            "action": action,
            "inputs": inputs
        }

    def assistant_decompose(self, assistant):
        if not assistant.find("Thought"):
            return -1
        return self.parse_output(assistant)

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

    def bm25_retrieve_query(self, query, top_k=5):
        return self.bm25_query.retrieve(query, top_k)

    def bm25_retrieve_thought(self, query, top_k=5):
        return self.bm25_thought.retrieve(query, top_k)

    def retrieve_action(self, action):
        return self.tool_doc_database[action]

    def get_conversation(self, i, j):
        return self.data[i]["conversations"][j]
    
    def get_conversations(self, i):
        return self.data[i]["conversations"]
    
    def retrieve_thought_ICL_after_query(self, query, top_k=5):
        thought_ICL = ""
        index = 1
        indices = self.bm25_retrieve_query(query, top_k=top_k)
        for indice in indices:
            i, j = self.query_database[indice][0], self.query_database[indice][1]
            conversations = self.get_conversations(i)
            example_query = self.get_conversation(i, j)['value']
            example_messages = [{'role': 'user', 'content': example_query}]
            assistant = self.assistant_decompose((self.get_conversation(i, j+1)['value']))
            if not isinstance(assistant, dict):
                continue
            thought = assistant['thought']
            thought = f"{DEFAULT_THOUGHT_TOKEN}{thought}"
            example_messages.append({'role': 'assistant', 'content': thought})
            example_ICL = self.get_prompt(example_messages)
            example_ICL = f"{index}. {example_ICL}\n"
            thought_ICL += example_ICL
            index += 1
        return thought_ICL

    def retrieve_thought_ICL(self, action, top_k=5):
        thought_ICL = ""
        index = 1
        action = action[action.find(".")+1:]
        try:
            action_indices = self.action_database[action]
        except:
            return -1
        if len(action_indices) > top_k:
            random.shuffle(action_indices)
        for action_indice in action_indices:
            i, j = action_indice
            example_conversation = self.get_conversation(i, j)["value"]
            assistant = self.assistant_decompose(example_conversation)
            if assistant["action"] == "Finish":
                continue
            try:
                next_conversation = self.get_conversation(i, j+1)["value"]
            except:
                continue
            try:
                next_assistant = self.assistant_decompose(self.get_conversation(i, j+2)["value"])
                if not isinstance(next_assistant, dict):
                    continue
            except:
                continue
            observation = f"\n{DEFAULT_OBSERVATION_TOKEN}{next_conversation}"
            example_conversation = example_conversation+observation
            next_thought = next_assistant["thought"]
            next_thought = f"{DEFAULT_THOUGHT_TOKEN}{next_thought}"
            example_messages = [{'role': 'user', 'content': example_conversation},
                                {'role': 'assistant', 'content': next_thought}]
            example_ICL = self.get_prompt(example_messages)
            example_ICL = f"{index}. {example_ICL}\n"
            thought_ICL += example_ICL
            index += 1
            if index > top_k:
                break
        
        return thought_ICL

            
            
            
    
    def retrieve_action_ICL(self, thought, top_k=5):
        action_ICL = ""
        index = 1
        indices = self.bm25_retrieve_thought(thought, top_k=top_k)
        for indice in indices:
            i, j = self.thought_database[indice][0], self.thought_database[indice][1]
            conversations = self.get_conversations(i)
            example_assistant = self.get_conversation(i, j)['value']
            assistant = self.assistant_decompose(example_assistant)
            if not isinstance(assistant, dict):
                continue
            example_thought = assistant['thought']
            example_thought = f"{DEFAULT_THOUGHT_TOKEN}{example_thought}"
            example_action = assistant['action']
            example_action = f"{DEFAULT_ACTION_TOKEN}{example_action}"
            example_messages = [{'role': 'user', 'content': example_thought},
                                {'role': 'assistant', 'content': example_action}]
            example_ICL = self.get_prompt(example_messages)
            example_ICL = f"{index}. {example_ICL}\n"
            action_ICL += example_ICL
            index += 1
        return action_ICL
    #通过搜出来的indice访问query/thought_database得到i，j，再通过i, j定位到对话历史的位置