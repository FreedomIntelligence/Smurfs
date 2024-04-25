# — coding: utf-8 –
import json
import sys
import argparse
import time
import requests
import os
from utils import change_name, standardize, get_white_list, get_answer_log, get_observation_log, build_tree, get_answer_details, test_sets
from tqdm import tqdm
# from langchain.llms import HuggingFacePipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from mistral import Mistral_Model
# from openai_llm import OpenAI_Model
from Smurfs.model.vllm_model.vllm_model import vllm_Model
from Smurfs.inference.server import get_rapidapi_response
# from Prompts.decompose_prompt import tool_check_prompt, answer_generation_direct_prompt, generate_subtask_prompt, choose_API_prompt, choose_parameter_prompt, answer_generation_prompt, final_answer_check_prompt, final_answer_generation_prompt, task_decompose_prompt, choose_tool_prompt
# from Prompts.no_easy_tool_prompt import tool_check_prompt, answer_generation_direct_prompt, generate_subtask_prompt, choose_parameter_prompt, answer_generation_prompt, final_answer_check_prompt, final_answer_generation_prompt, task_decompose_prompt, choose_tool_prompt
import threading
from Smurfs.agents.answer_agent.answer import answer_agent
from Smurfs.agents.executor_agent.executor import executor_agent
from Smurfs.agents.planning_agent.planner import planning_agent
from Smurfs.agents.verifier_agent.verifier import verifier_agent

def _Call_function(category, tool_name, api_name, tool_input, strip, white_list, toolbench_key):
    use_rapidapi_key = os.environ.get("use_rapidapi_key")
    rapidapi_key = os.environ.get("rapidapi_key")
    api_customization = os.environ.get("api_customization")

    api_name = change_name(standardize(api_name))
    tool_name = standardize(tool_name)
    if tool_name not in white_list.keys():
        print(f"tool name doesn't exist: {tool_name}")
        return {}, 1
    standard_tool_name = white_list[tool_name]["standard_tool_name"]
    payload = {
                    "category": category,
                    "tool_name": standard_tool_name,
                    "api_name": api_name,
                    "tool_input": tool_input,
                    "strip": strip,
                    "toolbench_key": toolbench_key
                }
    if use_rapidapi_key or api_customization:
        payload["rapidapi_key"] = rapidapi_key
        response = get_rapidapi_response(payload, api_customization=api_customization)
    else:
        time.sleep(2) # rate limit: 30 per minute
        headers = {"toolbench_key": toolbench_key}
        print(payload)
        # if tool_input == {}:
        #     response = requests.post("http://8.218.239.54:8080/rapidapi", headers=headers, timeout=15)
        # else:
        response = requests.post("http://8.218.239.54:8080/rapidapi", json=payload, headers=headers, timeout=15)
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
    elif response["error"] != "":
        status_code = "unexpected error, try again!"
    else:
        status_code = 0
    return json.dumps(response), status_code

def Call_function(category, tool_name, api_name, tool_input, strip, white_list):
    toolbench_key = os.environ.get("toolbench_key")
    response, status_code = _Call_function(category, tool_name, api_name, tool_input, strip, white_list, toolbench_key)
    if status_code == "unexpected error, try again!":
        arg = {change_name(k.lower()): v for k, v in tool_input.items()}
        response, status_code = _Call_function(category, tool_name, api_name, arg, strip, white_list, toolbench_key)
        if status_code == "unexpected error, try again!":
            arg = {change_name(k.replace("-", "_")): v for k, v in tool_input.items()}
            response, status_code = _Call_function(category, tool_name, api_name, arg, strip, white_list, toolbench_key)
            if status_code == "unexpected error, try again!":
                arg = {change_name(k.replace("\\", "")): v for k, v in tool_input.items()}
                response, status_code = _Call_function(category, tool_name, api_name, arg, strip, white_list, toolbench_key)
                if status_code == "unexpected error, try again!":
                    print(f"Call function fails")
                    with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                        line = json.dumps({
                            "id": 0,
                            "parameters": arg,
                            "wrong": response
                        }, ensure_ascii=False)
                        f.write(line + '\n')
                    return -1
    return response

def inference(query, relevant_APIs, white_list, subtask, chat, query_id, max_step=3):
    #index是工具路径列表，用在call_function的时候使用
    #tool_doc是工具的描述文档，用来查询使用工具的easytool描述
    #api_list是query文档里面的api_list，用来查询category name
    #tool_dic是query文档里面的tool_dic
    #这个函数执行对一个question的推理
    tool_check_num = Answer_Agent.run(question=query, task="tool_check", query_id=query_id)
    #直接回答问题
    if tool_check_num == 1:
        input_dic = {"task": query}
        answer = Answer_Agent.run(input_dic)
        return answer, answer, None, None
    # for i in range(len(tool_dic)):
    #     tool_dic[i]["Description"] = tool_dic[i]["Description"].split("\nThis tool has")[0]

    previous_log = []
    #用来生成半成品答案
    history_log = []
    tool_used_dic = {}
    relevant_APIs_ids = []
    for idx in relevant_APIs:
        ele = relevant_APIs[idx]
        relevant_APIs_ids.append(str(ele["ID"]))
    restart_time = 0
    step_num = 0
    hint = "Beginnig of the agent. No hint yet"
    retry_tool_id = 0
    #parameter生成有问题，尝试生成4次parameter，都不能生成正确的parameter那么parameter就是-1
    retry_parameter = 0
    re_time = 0
    subtask_id = 0
    restart = 0
    while True:
        if step_num >= max_step:
            print("\n\nReach steps limits, return answers!\n\n")
            answer_log = get_answer_log(history_log)
            answer = Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
            return answer, previous_log, re_time, history_log
        
        if step_num not in tool_used_dic.keys():
            tool_used_dic[step_num] = []

        tool_used = tool_used_dic[step_num]

        tool_list = []
        for idx in relevant_APIs:
            ele = relevant_APIs[idx]
            ID = str(ele['ID'])
            if ID in tool_used:
                continue
            des = ele['description']
            name = ele["tool_name"]
            tool_list.append({"ID": ID, "tool_name": name, "description": des})
            # tool_list.append(f'''ID: {ID}\n{des}''')

        if len(tool_list) == 0:
            if len(previous_log) == 0:
                answer_log = get_answer_log(history_log)
                partial_answer = Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
                answer = f"Sorry, I can't answer this question accurately using the existing tools. A partial answer is: {partial_answer}"
                return answer, previous_log, re_time, history_log
            else:
                delete_log = previous_log.pop()
                tool_used_dic[step_num] = []
                step_num -= 1
                tool_used_dic[step_num].append(delete_log["tool"])
                restart_time += 1
                re_time += 1
                continue

        current_log = {"thought": "", "action": "", "action_input": {}, "observation": "", "answer": "", "tool": "","id": subtask_id}

        answer_log = get_answer_log(previous_log)
       
        if retry_tool_id == 4:
            tool_id = tool_list[0]["ID"]
            tool_list = tool_list[0]
            thought = Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")

        else:
            thought = Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")
            tool_id = Executor_Agent.run(question=subtask, tool_list=tool_list, thought=thought, query_id=query_id, task="tool")
        
        try:
            #确保tool_id是个字符串
            tool_id = int(tool_id)
            tool_id = str(tool_id)
            if tool_id not in relevant_APIs_ids:
                re_time += 1
                retry_tool_id += 1
                print("Tool ID wrong! Generate tool_id that do not exist!")
                continue
            tool_des_json = relevant_APIs[str(tool_id)]
            retry_tool_id = 0
        except:
            retry_tool_id += 1
            print("Tool ID wrong! Generate tool_id that do not exist!")
            continue
        
        tool_name_list = tool_des_json["tool_name"].split(":")
        category_name = tool_name_list[0]
        tool_name = tool_name_list[1]
        api_name = tool_name_list[2]
        #只加入relevant api
        # api_name_list = []
        # for relevant_API in relevant_APIs:
        #     if relevant_API[0] == tool_name:
        #         api_name_list.append(relevant_API[1])
        API_doc = tool_des_json
        # API_doc = {}
        # for _api_name in tool_des_json["tool_guidelines"]:
        #     if _api_name in api_name_list:
        #         API_doc[_api_name] = tool_des_json["tool_guidelines"][_api_name]
        
        while True:
            try:
                parameters = {}

                if retry_parameter == 4:
                    restart = 1
                    retry_parameter = 0
                    print("No Para! Restart!")
                    break
                
                parameter = Executor_Agent.run(api_dic=API_doc, question=query, previous_log=answer_log, thought=thought, query_id=query_id, task="parameter")
                if parameter == -1:
                    retry_parameter += 1
                    re_time += 1
                    continue

                if parameter == {}:
                    retry_parameter = 0
                    parameters = {}
                    break

                for key in parameter:
                    value = parameter[key]
                    key = change_name(key)
                    parameters[key] = value

                retry_parameter = 0
                break

            except:
                if retry_parameter == 4:
                    parameters = {}
                    retry_parameter = 0
                    restart = 1
                    break
                retry_parameter += 1
                print("parameter generation fails, try again!")
                re_time += 1
                continue
    
        
        api_name = change_name(standardize(api_name))

        if restart != 1:
            try:
                observation = Call_function(category_name, tool_name, api_name, parameters, "truncate", white_list)
            except:
                observation = -1
                
            if observation == -1:
                restart = 1
                observation = str({"error": "", "response": "call API fails"})

        if restart == 1:
            tool_used_dic[step_num].append(str(tool_id))
            print('****Try Again For This Step****')
            re_time += 1
            restart = 0
            continue
        

        if len(previous_log) != 0:
            previous_id = previous_log[-1]["id"]
        else:
            previous_id = -1

        current_log["tool"] = str(tool_id)
        current_log["thought"] = thought
        current_log["action"] = api_name
        current_log["action_input"] = parameters
        current_log["observation"] = observation
        previous_log.append(current_log)

        observation_log = get_observation_log(previous_log)

        answer = Answer_Agent.run(question=subtask, call_result=observation_log, query_id=query_id, task="answer")

        previous_log[-1]["answer"] = answer

        history_log_ele = {"thought": thought, "action": tool_name, "action_input": parameters, "observation": observation, "answer": answer, "previous_id": previous_id, "id": subtask_id}
        history_log.append(history_log_ele)
        subtask_id += 1

        speak, status = Verifier_Agent.run(question=subtask, answer=answer, query_id=query_id)
        if speak == -1 and status == -1:
            step_num += 1
            continue

        try:
            if int(status) == 0:
                hint = speak
                step_num += 1
                continue
        except:
                step_num += 1
                continue
            
        else:
            return answer, previous_log, re_time, history_log

def decompose_inference(query, relevant_APIs, api_list, white_list, chat, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent, query_id):
    while True:
        subtasks = Planning_Agent.run(question=query, query_id=query_id)
        if subtasks == -1:
            continue
        break
    task_log = ""
    history_log = []
    previous_log_totals = []
    re_time_total = 0
    print(subtasks)
    relevant_API_list = {}
    tool_id = 0
    for api in api_list:
        for relevant_API in relevant_APIs:
            if relevant_API[0] == api["tool_name"] and relevant_API[1] == api["api_name"]:
                new_tool_name = api["category_name"]+":"+api["tool_name"]+":"+api["api_name"]
                ele = {"ID": tool_id, "tool_name": new_tool_name, "description": api["api_description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"]}
                relevant_API_list[str(tool_id)] = ele
                tool_id += 1

    for subtask in subtasks:
        task_log += f"question: {subtask}\n"
        answer, previous_log, re_time, previous_log_total = inference(task_log, relevant_API_list, white_list, subtask, chat, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent, query_id)
        previous_log_totals.append(previous_log_total)
        print(answer)
        history_log += previous_log
        re_time_total += re_time
        task_log += f"answer: {answer}\n"
    # input_dic = {"question": query, "previous_log": task_log}
    final_answer = Answer_Agent.run(question=query, previous_log=task_log, task="final", query_id=query_id)
    return final_answer, history_log, task_log, re_time_total, previous_log_totals 

def test(query_json, white_list, output_dir, chat, whole_solution_dir, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent):
    while True:
        try:
            global lock
            total_query = len(query_json)
            with tqdm(total=total_query, desc="Processing files", initial=0) as pbar:
                for i, test_query in enumerate(query_json, start=0):
                    # test_query = query_json[0]
                    idx = test_query[0]
                    test_query = test_query[1]
                    query = test_query["query"]
                    relevant_APIs = test_query["relevant APIs"]
                    api_list = test_query["api_list"]
                    final_answer, previous_log, task_log,re_time, previous_log_totals = decompose_inference(query, relevant_APIs, api_list, white_list, chat, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent, idx)
                    answer_details, total_steps = get_answer_details(final_answer, previous_log)
                    solution_tree, solution_total_steps = build_tree(previous_log_totals, previous_log, task_log)
                    output_file_ele = {
                        "query": query,
                        "restart_time": re_time,
                        "answer": {
                            "method": "decompose_dfs",
                            "total_steps": total_steps,
                            "final_answer": final_answer,
                            "answer_details": answer_details
                        }
                    }
                    
                    solution_file_ele = {
                        "query": query,
                        "total_steps": solution_total_steps,
                        "task_log": task_log,
                        "final_answer": final_answer,
                        "answer_path": answer_details,
                        "total_path": solution_tree
                    }
                    file_name = f"{idx}.json"
                    output_file = os.path.join(output_dir, file_name)
                    whole_solution_file = os.path.join(whole_solution_dir, file_name)
                    lock.acquire()
                    with open(output_file, "w") as file:
                        json.dump(output_file_ele, file, ensure_ascii=False, indent=4)
                    with open(whole_solution_file, "w") as file:
                        json.dump(solution_file_ele, file, ensure_ascii=False, indent=4)
                    lock.release()
                    pbar.update(1)
            return
                
        except Exception as e:
            print(e)
            print("some error occurs, continue...")
            time.sleep(60)
            continue

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_query_id_path', type=str, default="toolbench_data/data/test_query_ids", required=False, help='test query ids for different test sets')
    parser.add_argument('--method_name', type=str, default="smurfs-all", required=False, help='the inference method')
    parser.add_argument('--model_name', type=str, default="your_model_name", required=False, help='the model name for the vllm model')
    parser.add_argument('--query_file_dir', type=str, default="toolbench_data/data/test_instruction", required=False, help='the directory that contains test sets')
    # parser.add_argument("--tool_env_dir", action="store_true", help="Load lora model or not.")
    parser.add_argument('--tool_env_dir', type=str, default="toolbench_data/data/toolenv/tools", required=False, help='tool environment for the toolbench')
    parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    parser.add_argument('--api_customization', action="store_true", help="To use customized api or not.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    threads = []
    lock = threading.Lock()

    args = parse_arg()

    test_query_id_path = args.test_query_id_path
    method_name = args.method_name
    model_name = args.model_name
    query_file_dir = args.query_file_dir
    tool_env_dir = args.tool_env_dir

    toolbench_key = args.toolbench_key
    rapidapi_key = args.rapidapi_key
    use_rapidapi_key = args.use_rapidapi_key
    api_customization = args.api_customization
    

    os.environ['toolbench_key'] = toolbench_key
    os.environ['rapidapi_key'] = rapidapi_key
    os.environ['use_rapidapi_key'] = use_rapidapi_key
    os.environ['api_customization'] = api_customization
    # test_query_id_path = "/mntcephfs/lab_data/chenjunzhi/data/test_query_ids"
    # method_name = "vicuna-7b-no-easytool"
    # model_name = "vicuna-7b-v1.1"
    # query_file_dir = "/mntcephfs/lab_data/chenjunzhi/data/test_instruction"
    # # tool_doc_dir = '/Users/chenjunzhi/Downloads'
    # tool_env_dir = "/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools"

    chat = vllm_Model(model_name)
    

    for test_set in test_sets:
        total_output_file = f"data/{method_name}/{test_set}_raw.json"
        
        test_ids = list(json.load(open(os.path.join(test_query_id_path, test_set+".json"), "r")).keys())

        query_file = f'{query_file_dir}/{test_set}.json'
        
        output_dir = f"data/{method_name}/{test_set}/answer"

        whole_solution_dir = f"data/{method_name}/{test_set}/whole"

        logger_dir = f"data/{method_name}/{test_set}/agent_log"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(whole_solution_dir):
            os.makedirs(whole_solution_dir)
        
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        Answer_Agent = answer_agent(chat, logger_dir)
        Executor_Agent = executor_agent(chat, logger_dir)
        Planning_Agent = planning_agent(chat, logger_dir)
        Verifier_Agent = verifier_agent(chat, logger_dir)

        items = os.listdir(output_dir)
        for i in range(len(items)):
            items[i] = items[i].split(".")[0]

        white_list = get_white_list(tool_env_dir)

        with open(query_file) as file:
            query_json = json.load(file)
        # with open(tool_doc_dir) as file:
        #     tool_doc = json.load(file)
        
        print(len(items))
        query_json_to_do = []
        # if len(items) != 0:
        for idx, q in enumerate(query_json):
            # print(idx)
            if str(idx) in items:
                continue
            query_id = q["query_id"]
            if str(query_id) not in test_ids:
                continue
            query_json_to_do_ele = (idx, q)
            query_json_to_do.append(query_json_to_do_ele)
        # else:
        #     query_json_to_do = query_json
        
        total_len = len(query_json_to_do)
        query_len = len(query_json)
        print(total_len)

        if total_len < 20:
            for i in range(total_len):
                if total_len == 0:
                    break
            
                start = i
                end = i+1
                if i == total_len-1:
                    query_json_cur = query_json_to_do[start:]
                else:
                    query_json_cur = query_json_to_do[start: end]
                t = threading.Thread(target=test, args=(query_json_cur, white_list, output_dir, chat, whole_solution_dir, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent))
                t.start()
                threads.append(t)
            
                
        else:
            for i in range(20):        
                
                if total_len == 0:
                    break
            
                start = round(total_len/20)*i
                end = round(total_len/20)*(i+1)
                if i == 19:
                    query_json_cur = query_json_to_do[start:]
                else:
                    query_json_cur = query_json_to_do[start: end]
                t = threading.Thread(target=test, args=(query_json_cur, white_list, output_dir, chat, whole_solution_dir, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent))
                t.start()
                threads.append(t)
        
        for thread in threads:
            thread.join()

        total_json = {}
        items = os.listdir(output_dir)
        for item in items:
            item_path = os.path.join(output_dir, item)
            idx = item.split(".")[0]
            total_json[str(idx)] = json.load(open(item_path, 'r'))

        with open(total_output_file, 'w') as file:
            json.dump(total_json, file, indent=4, ensure_ascii=False)
    