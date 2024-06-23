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

class function_call_executor_agent(BaseAgent):
    thought_prompt: Any
    tool_prompt: Any
    parameter_prompt: Any
    functions: list
    api_name_reflect: dict
    tool_names: list
    cate_names: list
    max_observation_length: int
    use_rapidapi_key: bool
    api_customization: bool
    toolbench_key: str
    service_url: str

    def __init__(self, *args, **kwargs):
        thought_prompt = generate_thought_prompt
        tool_prompt = choose_tool_prompt
        parameter_prompt = choose_parameter_prompt
        name = "Executor Agent"
        kwargs.update({"thought_prompt": thought_prompt})
        kwargs.update({"tool_prompt": tool_prompt})
        kwargs.update({"parameter_prompt": parameter_prompt})
        kwargs.update({"name": name})
        kwargs.update({"functions": []})
        kwargs.update({"api_name_reflect": {}})
        kwargs.update({"tool_names": []})
        kwargs.update({"cate_names": []})
        super().__init__(
            *args,
            **kwargs,
        )

    def run(self, query_id, task, **kwargs):
        """agent run one step"""
        api_list = kwargs["api_list"]
        white_list = kwargs["white_list"]
        data_dict = self.fetch_api_json(api_list)
        origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
        tool_des = contain(origin_tool_names,white_list)
        if tool_des == False:
            pass
        tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
        for k,api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_des[k][0]
            openai_function_json,cate_name, pure_api_name = self.api_json_to_openai_json(api_json,standard_tool_name)
            self.functions.append(openai_function_json)

            self.api_name_reflect[openai_function_json["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)
        
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
                    print(result)
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
            tools = kwargs["tools"]
            ind = 0
            while True:
                try:
                    result = self.llm.prediction(message, tools=tools)
                    tool_name = result['tool_calls'][0]["function"]['name']
                    tool_input = eval(result['tool_calls'][0]["function"]["arguments"])
                    # obs, code = self.step(action_name=tool_name, action_input=tool_input)
                    return tool_name, tool_input
                except Exception as e:
                    print(f"choosing tool fails: {e}")
                    self.log(query_id, f"choosing tool fails: {e}")
                    if ind > self.max_retry:
                        return -1, -1
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
            agent_prompt = self.memory["question"]
            
        return agent_prompt

    def fetch_api_json(self, api_list):
        data_dict = {"api_list":[]}
        for item in api_list:
            cate_name = item["category_name"]
            tool_name = standardize(item["tool_name"])
            api_name = change_name(standardize(item["api_name"]))
            tool_json = json.load(open(os.path.join(self.tool_root_dir, cate_name, tool_name + ".json"), "r"))
            append_flag = False
            api_dict_names = []
            for api_dict in tool_json["api_list"]:
                api_dict_names.append(api_dict["name"])
                pure_api_name = change_name(standardize(api_dict["name"]))
                if pure_api_name != api_name:
                    continue
                api_json = {}
                api_json["category_name"] = cate_name
                api_json["api_name"] = api_dict["name"]
                api_json["api_description"] = api_dict["description"]
                api_json["required_parameters"] = api_dict["required_parameters"]
                api_json["optional_parameters"] = api_dict["optional_parameters"]
                api_json["tool_name"] = tool_json["tool_name"]
                data_dict["api_list"].append(api_json)
                append_flag = True
                break
            if not append_flag:
                print(api_name, api_dict_names)
        return data_dict

    def api_json_to_openai_json(self, api_json,standard_tool_name):
        description_max_length=256
        templete =     {
            "name": "",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [],
                "optional": [],
            }
        }
        
        map_type = {
            "NUMBER": "integer",
            "STRING": "string",
            "BOOLEAN": "boolean"
        }

        pure_api_name = change_name(standardize(api_json["api_name"]))
        templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
        templete["name"] = templete["name"][-64:]

        templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
        
        if api_json["api_description"].strip() != "":
            tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
            templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
        if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
            for para in api_json["required_parameters"]:
                name = standardize(para["name"])
                name = change_name(name)
                if para["type"] in map_type:
                    param_type = map_type[para["type"]]
                else:
                    param_type = "string"
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                }

                default_value = para['default']
                if len(str(default_value)) != 0:    
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length],
                        "example_value": default_value
                    }
                else:
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length]
                    }

                templete["parameters"]["properties"][name] = prompt
                templete["parameters"]["required"].append(name)
            for para in api_json["optional_parameters"]:
                name = standardize(para["name"])
                name = change_name(name)
                if para["type"] in map_type:
                    param_type = map_type[para["type"]]
                else:
                    param_type = "string"

                default_value = para['default']
                if len(str(default_value)) != 0:    
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length],
                        "example_value": default_value
                    }
                else:
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length]
                    }

                templete["parameters"]["properties"][name] = prompt
                templete["parameters"]["optional"].append(name)

        return templete, api_json["category_name"],  pure_api_name

    def step(self,**args):
        obs, code = self._step(**args)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + "..."
        return obs, code

    def _step(self, action_name="", action_input=""):
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """
        if action_name == "Finish":
            try:
                json_data = json.loads(action_input,strict=False)
            except:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer
            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                
                # self.success = 1 # succesfully return final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                return "{error:\"\"return_type\" is not a valid choice\"}", 2
        else:
            for k, function in enumerate(self.functions):
                if function["name"].endswith(action_name):
                    pure_api_name = self.api_name_reflect[function["name"]]
                    payload = {
                        "category": self.cate_names[k],
                        "tool_name": self.tool_names[k],
                        "api_name": pure_api_name,
                        "tool_input": action_input,
                        "strip": self.observ_compress_method,
                        "toolbench_key": self.toolbench_key
                    }
                    # if self.process_id == 0:
                    #     print(colored(f"query to {self.cate_names[k]}-->{self.tool_names[k]}-->{action_name}",color="yellow"))
                    if self.use_rapidapi_key or self.api_customization:
                        payload["rapidapi_key"] = self.rapidapi_key
                        response = get_rapidapi_response(payload, api_customization=self.api_customization)
                    else:
                        time.sleep(2) # rate limit: 30 per minute
                        headers = {"toolbench_key": self.toolbench_key}
                        response = requests.post(self.service_url, json=payload, headers=headers, timeout=15)
                        if response.status_code != 200:
                            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                        try:
                            response = response.json()
                        except:
                            print(response)
                            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1

