# — coding: utf-8 –
import openai
import json
import logging
import sys
import argparse
import time
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import numpy as np
import requests
import os
import subprocess
import re
import importlib.util
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from utils import *
from tqdm import tqdm
# from langchain.llms import HuggingFacePipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mistral import Mistral_Model
from openai_llm import OpenAI_Model
# from Prompts.decompose_prompt import tool_check_prompt, answer_generation_direct_prompt, generate_subtask_prompt, choose_API_prompt, choose_parameter_prompt, answer_generation_prompt, final_answer_check_prompt, final_answer_generation_prompt, task_decompose_prompt, choose_tool_prompt
from Prompts.no_easy_tool_prompt import tool_check_prompt, answer_generation_direct_prompt, generate_subtask_prompt, choose_parameter_prompt, answer_generation_prompt, final_answer_check_prompt, final_answer_generation_prompt, task_decompose_prompt, choose_tool_prompt



def build_index(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name not in index:
                index[dir_name] = []
            index[dir_name].append(root)
    return index

chat = Mistral_Model("/mntcephfs/lab_data/chenjunzhi/mistralai:Mistral-7B-Instruct-v0.2")
# chat = OpenAI_Model("gpt-3.5-turbo-0613")

def tool_check(task, chat):
    prompt = tool_check_prompt.format(task=task)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            result = eval(result)
            a = result["Reason"]
            b = result["Choice"]
            # print(a)
            if 'yes' in b.lower():
                return result, -1
            else:
                return result, 1
        except Exception as e:
            print(f"tool check fails: {e}")
            if ind > 10:
                return "", -1
            ind += 1
            continue

# task = "Who is Obama"
# result, check_index = tool_check(task)
# print(result)
# print(check_index)

def choose_tool(question, tool_list, thought):
    question = question+f"thought: {thought}\n"
    prompt = choose_tool_prompt.format(question=question, Tool_list=tool_list)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            start = result.find("{")
            end = result.rfind("}")
            result = eval(result[start:end+1])
            print(result)
            tool = result['ID']
            # print(a)
            return tool
        except Exception as e:
            print(f"generating subtask fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue

# tool_list = ["ID: 917\n'Orderful' is an API tool which can handle EDI transactions. \nThis tool has 1 API:\n1. 'Transactions' can get a transaction by ID.", "ID: 913\n'suivi-colis' can track packages in Nouvelle-Calédonie.\nThis tool has 4 APIs:\n1. 'Health' can get the API's health.\n2. 'Latest' can retrieve the current status of a package.\n3. 'Count' can count the number of package status updates, useful for limiting network consumption or IoT resources.\n4. 'All' can retrieve the entire package history from its shipment to the current status."]
# thought = """Based on the user's request, I would suggest using the 'suivi-colis' tool with the API 'Latest' to obtain the most recent status information for both packages mentioned in the user's query - a1890 and CA105408006SI. This should provide us with the necessary details to update the user about their packages. If there have been multiple status updates for either package the user's limit, we could consider using the 'Count' API to determine the number of updates before making additional requests. In case of any issues or errors during this process, I will refer back to the 'Health' API to check if there might be any problems with the service itself"""
# question = "Use the 'suivi-colis' tool with the Latest API to check the latest status of both packages with the provided references a1890 and CA105408006SI."

# tool = choose_tool(question, tool_list, thought)

def answer_generation_direct(task, chat):
    prompt = answer_generation_direct_prompt.format(task=task)
    message = [{'role': 'user',
                'content': prompt}]
    result = chat.prediction(message)
    return result

# task = "Who is Obama"
# result = answer_generation_direct(task, chat)
# print(result)

def task_decompose(question, chat):
    prompt = task_decompose_prompt.format(question=question)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            # print(result)
            start = result.find("{")
            end = result.find("}")
            result = eval(result[start:end+1])
            # print(result)
            subtasks = result['Tasks']
            # print(a)
            return subtasks
        except Exception as e:
            print(f"task deompose fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue


# query = "My friends and I are eagerly awaiting the delivery of a package. Can you please track the package with the Pack & Send reference number a1890? Additionally, I'm interested in the latest status of the package with colis ID 'CA105408006SI'."
# subtasks = task_decompose(query, chat)

def generate_thought(question, tool_list, previous_log, hint, chat):
    prompt = generate_subtask_prompt.format(task=question, functions=tool_list, messages=previous_log, hint=hint)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            print(result)
            return result
        except Exception as e:
            print(f"generating subtask fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue

# tool_dic = [
#              {
#                 "ID": 917,
#                 "Description": "'Orderful' is an API tool which can handle EDI transactions. \nThis tool has 1 API:\n1. 'Transactions' can get a transaction by ID."
#             },
#             {
#                 "ID": 913,
#                 "Description": "'suivi-colis' can track packages in Nouvelle-Calédonie.\nThis tool has 4 APIs:\n1. 'Health' can get the API's health.\n2. 'Latest' can retrieve the current status of a package.\n3. 'Count' can count the number of package status updates, useful for limiting network consumption or IoT resources.\n4. 'All' can retrieve the entire package history from its shipment to the current status."
#             }
#         ]
# question = "Check the latest status of the package with Colis ID CA105408006SI."
# previous_log = ""
# thought = generate_thought(question, tool_dic, [], previous_log, "", chat)
# print(thought)
        
#response: Based on the user's request, I would suggest using the 'suivi-colis' tool with the API 'Latest' to obtain the most recent status information for both packages mentioned in the user's query - a1890 and CA105408006SI. This should provide us with the necessary details to update the user about their packages. If there have been multiple status updates for either package the user's limit, we could consider using the 'Count' API to determine the number of updates before making additional requests. In case of any issues or errors during this process, I will refer back to the 'Health' API to check if there might be any problems with the service itself.


def choose_parameter_depend(api_dic, question, previous_log, thought, chat):
    question = question+f"thought: {thought}\n"
    if len(api_dic["required_parameters"]) == 0 and len(api_dic["optional_parameters"]) == 0:
        return {}
    prompt = choose_parameter_prompt.format(api_dic=api_dic, previous_log=previous_log, question=question)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            start = result.find("{")
            end = result.rfind("}")
            print(result[start:end+1])
            result = result[start:end+1]
            clean_answer = eval(
                result.replace(": true", ": True").replace(": false", ": False").replace("```", "").strip())
            # a = clean_answer["Parameters"]
            clean_answer = clean_answer["Parameters"]
            return clean_answer
        except Exception as e:
            print(f"choose parameter depend fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue


# api_dic = {
#                 "name": "Latest",
#                 "description": "L'état courant (ie. le dernier état du colis).",
#                 "required_parameters": [
#                     {
#                         "name": "colisId",
#                         "type": "string",
#                         "description": "",
#                         "default": "CA107308006SI"
#                     }
#                 ],
#                 "optional_parameters": [],
#                 "Example": {
#                     "Scenario": "if you want to get the latest status of package with ID CA107308006SI",
#                     "Parameters": {
#                         "colisId": "CA107308006SI"
#                     }
#                 }
#             }

# thought = """Based on the user's request, I would suggest using the 'suivi-colis' tool with the API 'Latest' to obtain the most recent status information for both packages mentioned in the user's query - a1890 and CA105408006SI. This should provide us with the necessary details to update the user about their packages. If there have been multiple status updates for either package the user's limit, we could consider using the 'Count' API to determine the number of updates before making additional requests. In case of any issues or errors during this process, I will refer back to the 'Health' API to check if there might be any problems with the service itself"""

# question = "Use the 'suivi-colis' tool with the Latest API to check the latest status of both packages with reference numbers a1890 and CA105408006SI."
# # previous_log = [{"task": "Execute the task step by step by decomposing it to a series of subtasks that can be solved using a single tool(functions): My friends and I are eagerly awaiting the delivery of a package. Can you please track the package with the Pack & Send reference number a1890? Additionally, I'm interested in the latest status of the package with colis ID 'CA105408006SI'."}]
# para = choose_parameter_depend(api_dic, question, "", thought, chat)
# print(para)
#response: [{'colisId': 'a1890'}, {'colisId': 'CA105408006SI'}]


# todo: 加入原函数里面的去除异常情况的参数化尝试。
def _Call_function(category, tool_name, api_name, tool_input, strip, white_list, toolbench_key="lDPxqwOYRltCORiQt4viGwgAehsDt2pn5ez4W78hwK5nVtyPET"):
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

def Call_function(category, tool_name, api_name, tool_input, strip, white_list, toolbench_key="lDPxqwOYRltCORiQt4viGwgAehsDt2pn5ez4W78hwK5nVtyPET"):
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
# white_list = get_white_list("/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools")
# toolbench_key = "lDPxqwOYRltCORiQt4viGwgAehsDt2pn5ez4W78hwK5nVtyPET"
# category = "Logistics"
# tool_name = "suivi-colis"
# api_name = "Latest"
# tool_input = {'colisid': 'CA105408006SI'}
# strip = "truncate"
# response  = Call_function(category, tool_name, api_name, tool_input, strip, white_list, toolbench_key)
# print(response)
# parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='observation compress method')
#response: {"error": "", "response": ""}
#改为response: {"error": "", "response": "arrived"}


# tool_name = "amazon_translate"
# api_name = "translate_using_query"
# index = build_index("/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools")
# arg = {"target_lang": "de", "text": "Good Morning!"}
# id = 1
# call_result = Call_function(tool_name, api_name, arg, index, id)
# print(call_result)

def answer_generation(question, call_result, chat):
    prompt = answer_generation_prompt.format(question=question, call_result=call_result)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            break
        except Exception as e:
            print(f"answer generation fails: {e}")
            if ind > 2:
                return -1
            ind += 1
            continue
    return result


# question = "Use the 'suivi-colis' tool with the Latest API to check the latest status of both packages with the provided references a1890 and CA105408006SI."
# call_result = [{"error": "", "response": "package with the provided references a1890 have arrived"}, {"error": "", "response": "package with the provided references CA105408006SI have arrived"}]
# answer = answer_generation(question, call_result, chat)
# print(answer)
#response: Based on the API response, I can confirm that both packages with references a1890 and CA105408006SI have arrived at their destination.

# {"Parameters":[{"colisId":"CA107308006SI"},{"colisId":"ReferenceNumberHere"}]}

# def answer_check(question, answer, chat):
#     prompt = answer_check_prompt.format(question=question, answer=answer)
#     message = [{'role': 'user', 
#                  'content': prompt}]
#     ind = 0
#     while True:
#         try:
#             result = chat.prediction(message)
#             print(result)
#             break
#         except Exception as e:
#             print(f"answer check fails: {e}")
#             if ind > 2:
#                 return -1
#             ind += 1
#             continue
#     if 'yes'.lower() in str(result).lower():
#         return 1
#     else:
#         try:
#             result = eval(result)
#             return result["Reason"]
#         except:
#             return result
    
# question = "Use the 'suivi-colis' tool with the Latest API to check the latest status of both packages with the provided references a1890 and CA105408006SI."
# answer = "I can not get anything from the API response"
# answer_result = answer_check(question, answer, chat)
# print(answer_result)
#response
# {
# "Reason": "The response does not provide any information regarding the status check of the packages using the 'suivi-colis' tool with the Latest API.",
# "Choice": "No"
# }

def evaluate(question, answer, chat):
    prompt = final_answer_check_prompt.format(question=question, answer=answer)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            start = result.find("{")
            end = result.find("}")
            print(result)
            clean_result = eval(result[start:end+1])
            speak = clean_result["Speak"]
            status = clean_result["Status"]
            return speak, status
        except Exception as e:
            print(f"final answer check fails: {e}")
            if ind > 2:
                return -1, -1
            ind += 1
            continue

def final_answer_generate(question, previous_log, chat):
    prompt = final_answer_generation_prompt.format(question=question, previous_log=previous_log)
    message = [{'role': 'user', 
                 'content': prompt}]
    ind = 0
    while True:
        try:
            result = chat.prediction(message)
            break
        except Exception as e:
            print(f"answer generation fails: {e}")
            if ind > 2:
                return -1
            ind += 1
            continue
    return result

# question = "I am a music enthusiast and I need help finding the latest coupons for purchasing musical instruments. Can you provide me with the latest coupons and popular coupons? Additionally, I would like to explore the available sorting methods for products. Please provide me with the list of available sorting methods."
# previous_log = ""
# tool_list = """1.blocktrail_bitcoin_developers_platform: BlockTrail is a bitcoin API and developers platform, enabling data and payments for developers, enterprises and service providers\n2.halobiru_store: Online Shop HaloBiru.store\n3.tokopedia_super_api: Unleash the Power of Tokopedia: Effortlessly Retrieve Shop and Product Information with Our API!\n4.get_27coupons: 27coupons is one of the largest aggregators of coupons and deals data of India. We have been successfully running our website www.27coupons.com since July 2011. Our API framework offers access to data allowing developers to build upon and extend their applications in new and creative ways. Our APIs are constantly evolving and developing as per the industry standards.\rOur API framework is built upon REST architecture drawing inspiration from API frameworks of leading websites such as Twitter, Facebook and S"""
# subtask, tool = generate_subtask(question, previous_log, tool_list)
# print(subtask)
# print(tool)

def get_answer_log(log):
    if log == []:
        return "Beginnig of the agent. No log yet"
    answer_logs = []
    for ele in log:
        answer_log = {"thought": "", "answer": ""}
        answer_log["thought"] = ele["thought"]
        answer_log["answer"] = ele["answer"]
        answer_logs.append(answer_log)
    return answer_logs

def get_observation_log(log):
    if log == []:
        return ""
    answer_logs = []
    for i, ele in enumerate(log):
        if i == len(log)-1:
            answer_log = {"thought": "", "observation": ""}
            answer_log["thought"] = ele["thought"]
            answer_log["observation"] = ele["observation"]
            answer_logs.append(answer_log)
        else:
            answer_log = {"thought": "", "answer": ""}
            answer_log["thought"] = ele["thought"]
            answer_log["answer"] = ele["answer"]
            answer_logs.append(answer_log)
    return answer_logs

def inference(query, relevant_APIs, api_list, white_list, tool_doc, tool_dic, subtask, chat, restart_num=5, max_step=3):
    #index是工具路径列表，用在call_function的时候使用
    #tool_doc是工具的描述文档，用来查询使用工具的easytool描述
    #api_list是query文档里面的api_list，用来查询category name
    #tool_dic是query文档里面的tool_dic
    #这个函数执行对一个question的推理

    tool_check_num = tool_check(query, chat)
    #直接回答问题
    if tool_check_num == 1:
        answer = answer_generation_direct(query, chat)
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
            answer = final_answer_generate(query, answer_log, chat)
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
                partial_answer = final_answer_generate(query, answer_log, chat)
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
            thought = generate_thought(query, tool_list, answer_log, hint, chat)

        else:
            thought = generate_thought(query, tool_list, answer_log, hint, chat)
            tool_id = choose_tool(subtask, tool_list, thought)
        
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
                
                parameter = choose_parameter_depend(API_doc, query, answer_log, thought, chat)
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
            observation = Call_function(category_name, tool_name, api_name, parameters, "truncate", white_list)
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

        answer = answer_generation(subtask, observation_log, chat)

        previous_log[-1]["answer"] = answer

        history_log_ele = {"thought": thought, "action": tool_name, "action_input": parameters, "observation": observation, "answer": answer, "previous_id": previous_id, "id": subtask_id}
        history_log.append(history_log_ele)
        subtask_id += 1

        speak, status = evaluate(subtask, answer, chat)
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
            return speak, previous_log, re_time, history_log

            

def decompose_inference(query, relevant_APIs, api_list, white_list, tool_doc, tool_dic, chat, restart_num=5):
    while True:
        subtasks = task_decompose(query, chat)
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
        answer, previous_log, re_time, previous_log_total = inference(task_log, relevant_API_list, api_list, white_list, tool_doc, tool_dic, subtask, chat, restart_num)
        previous_log_totals.append(previous_log_total)
        print(answer)
        history_log += previous_log
        re_time_total += re_time
        task_log += f"answer: {answer}\n"
    final_answer = final_answer_generate(query, task_log, chat)
    return final_answer, history_log, task_log, re_time_total, previous_log_totals 

def build_tree(previous_log_totals, answer_path_log, task_log):
    
    total_root_list = []
    total_total_steps = 0
    task_log_list = task_log.split("question: ")[1:]
    for i in range(len(task_log_list)):
        task_log_list[i] = task_log_list[i].split("answer: ")
    
    for j, previous_log_total in enumerate(previous_log_totals):
        
        if previous_log_total == None:
            
            answer_detail = {
                "role": "plan_global",
                "message": {
                    "subtask": task_log_list[j][0],
                    "subtask_answer": task_log_list[j][1]
                },
                "total_steps": 0,
                "next": []
            }
            total_root_list.append(answer_detail)
            continue

        next_list = []
        root_list = []
        total_steps = 0
        for i in range(len(previous_log_total)):
            
            current_log = previous_log_total[i]
            
            tool_call_list = []
            api_name = current_log["action"]
            parameter = current_log["action_input"]
            response = current_log["observation"]
            next_ele = {
                "role": "tool",
                "message": {
                    "name": api_name,
                    "arguments": parameter,
                    "response": response
                },
                "next": []
            }
            tool_call_list.append(next_ele)
            total_steps += 1
            if len(tool_call_list) > 1:
                for k in range(len(tool_call_list)-2, -1, -1):
                    tool_call_list[k]["next"].append(tool_call_list[k+1])
            next_list.append(tool_call_list[0])
        total_total_steps += total_steps
        
        for i in range(len(next_list)-1, -1, -1):
            current_log = next_list[i]
            current_log_pre_id = previous_log_total[i]["previous_id"]
            if current_log_pre_id == -1:
                # print(current_log)
                root_list.append(current_log)
            else:
                next_list[current_log_pre_id]["next"].append(current_log)
        answer_detail = {
            "role": "plan_global",
            "message": {
                "subtask": task_log_list[j][0],
                "subtask_answer": task_log_list[j][1]
            },
            "total_steps": total_steps,
            "next": root_list
        }
        total_root_list.append(answer_detail)

    answer_details = {
        "role": "system",
        "message": "",
        "next": [
            {
                "role": "user",
                "message": "",
                "next": total_root_list
            }
        ]
    }
    return answer_details, total_total_steps


def get_answer_details(final_answer, previous_log):
   
    next_list = []
    total_steps = 0
    for i in range(len(previous_log)):
        current_log = previous_log[i]
        if not isinstance(current_log, dict):
            next_ele = {
                "role": "assistant",
                "message": current_log,
                "next": []
            }
            next_list.append(next_ele)
            total_steps += 1
            continue
       
        api_name = current_log["action"]
        parameter = current_log["action_input"]
        response = current_log["observation"]
        next_ele = {
            "role": "tool",
            "message": {
                "name": api_name,
                "arguments": parameter,
                "response": response
            },
            "next": []
        }
        next_list.append(next_ele)
        total_steps += 1
    answer_ele = {
        "role": "tool",
        "message": {
            "name": "Finish",
            "arguments": {
                "return_type": "give_answer",
                "final_answer": final_answer
            },
            "response": ""
        },
        "next": []
    }
    next_list.append(answer_ele)
    for i in range(len(next_list)-2, -1, -1):
        next_list[i]["next"].append(next_list[i+1])
    next_result = next_list[0]
    answer_details = {
        "role": "system",
        "message": "",
        "next": [
            {
                "role": "user",
                "message": "",
                "next": [next_result]
            }
        ]
    }
    return answer_details, total_steps

# with open("/home/chenjunzhi/JARVIS/easytool/data_toolbench/test_data/G2_category.json") as file:
#     query_json = json.load(file)
# test_query = query_json[0]
# print(test_query)
# query = test_query["query"]
# api_list = test_query["api_list"]
# white_list = get_white_list("/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools")
# with open("/home/chenjunzhi/JARVIS/easytool/data_toolbench/tool_instruction/toolbench_tool_instruction.json") as file:
#     tool_doc = json.load(file)
# tool_dic = test_query["Tool_dic"]
# relevant_apis = test_query["relevant APIs"]

# final_answer, previous_log, task_log, re_time_total, previous_log_totals = decompose_inference(query, relevant_apis,api_list, white_list, tool_doc, tool_dic, chat)
# print(final_answer)
# print("\n\n")
# print(task_log)
# print("\n\n")
# print(re_time_total)
# print("\n\n")
# print(previous_log)

# with open("Inference/previous_log.json", "w") as file:
#     json.dump(previous_log, file, indent=4)

# with open("Inference/history_log.json", "w") as file:
#     json.dump(previous_log_totals, file, indent=4)

# answer_details = get_answer_details(final_answer, previous_log)
# with open("Inference/answer.json", "w") as file:
#     json.dump(answer_details, file, indent=4)
# print("\n\n")
# solution_tree, total_steps = build_tree(previous_log_totals, previous_log, task_log)
# print(total_steps)
# with open("Inference/solution.json", "w") as file:
#     json.dump(solution_tree, file, indent=4)



# # # final_answer = """Firstly, regarding the postal code and district for the Istanbul province with plate number 34, according to the first subtask, there are multiple postal codes associated with this plate number, specifically in the districts of Adalar and Ilce. Therefore, depending on which neighborhood within these districts you are interested in, you can refer to the provided postal codes ('34975', '34970', '34973', or '34977'). However, since no specific neighborhood was mentioned in your query, I couldn't determine which postal code applies to your situation directly from the given information.

# # # Secondly, concerning your request for transit agencies in Istanbul along with their names and contact numbers, the second subtask yielded positive results. According to the execution log, various transit agencies were found, including EKVF, ACT – Agence Calédonienne de Transit, AFL – Agence de Fréight et Logistique, Agence Logistique et Transit Nouméa, ATI TRAMAR-Groupe Léon Vincent, and ATL Lines DEMLINES. Each transit agency's unique identifier, name, and contact details (phone number and website or email address) were also obtained through the API call. You may choose one or more of these agencies depending on your travel requirements and preferences."""

# # # previous_log = [{'task': "Execute the task step by step by decomposing it to a series of subtasks that can be solved using a single tool(functions): I'm planning a trip to Turkey and need information about postal codes in Istanbul. Can you provide me with the postal code and district for Istanbul province with plate number 34? Additionally, I would like to know if there are any transit agencies available in Istanbul. Please fetch their names and contact numbers."}, {'subtask': "Use the 'Turkey PostalCodes' tool to find the postal code associated with the given Istanbul plate number (34),", 'tool': '910', 'answer': "Based on the API response, there are multiple postal codes associated with the given Istanbul plate number (34). Specifically, for the il (province) of Istanbul and ilce (district) of Adalar, there are six different postal codes: '34975' for Burgazada Mahalle (neighborhood); '34970' for Buyukcedid and Nizam Mahalles; and '34973' and '34977' for Buyukkada and Kinaliada Mahalles, respectively.", 'current_final_answer': "Based on your query, I have completed a subtask where I used the 'Turkey PostalCodes' tool to search for the postal codes corresponding to the given Istanbul plate number (34). According to the obtained data, there are several postal codes related to this plate number within the Istanbul province. More specifically, the following districts and their respective postal codes were identified:\n- Adalar: '34975' for Burgazada Mahalle\n- Istanbul (unspecified district): '34970' for Buyukcedid and Nizam Mahalles\n- Buyukkada: '34973'\n- Kinaliada: '34977'\nRegarding your second request, I conducted another subtask aimed at discovering whether there exist transit agencies in Istanbul. As per my findings, yes, there are numerous public transportation providers operating in Istanbul. Here are some of the major ones along with their contact details:\n1. IETT - Istanbul Electric Tramway and Tunnel Company: Phone: +90 212 216 10 00\n2. MHD - Istanbul Metropolitan Municipality Bus Directorate: Phone: +90 212 518 11 11\n3. Marmaray: High-speed railway connecting Europe and Asia sides of Istanbul: Phone: +90 212 518 12 00\n4. IDO - Istanbul Seabuses Administration: Phone: +90 212 533 45 55\nThese agencies offer various modes of transport such as buses, trams, metro, funiculars, ferries, and trains to facilitate smooth travel around Istanbul.", 'api_response': [{'name': 'il', 'parameter': {'il': 34}, 'response': ['{"error": "", "response": "{\'success\': True, \'status\': \'ok\', \'dataupdatedate\': \'2023-07-04\', \'description\': \'PTT taraf\\u0131ndan g\\u00fcnl\\u00fck olarak \\u00e7ekilerek g\\u00fcncellenen posta kodlar\\u0131d\\u0131r.\', \'pagecreatedate\': \'2023-07-04 00:00:00\', \'postakodu\': [{\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ADALAR\', \'semt_bucak_belde\': \'BURGAZADA\', \'mahalle\': \'BURGAZADA MAH\', \'pk\': \'34975\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ADALAR\', \'semt_bucak_belde\': \'B\\u00dcY\\u00dcKADA\', \'mahalle\': \'MADEN MAH\', \'pk\': \'34970\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ADALAR\', \'semt_bucak_belde\': \'B\\u00dcY\\u00dcKADA\', \'mahalle\': \'N\\u0130ZAM MAH\', \'pk\': \'34970\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ADALAR\', \'semt_bucak_belde\': \'HEYBEL\\u0130ADA\', \'mahalle\': \'HEYBEL\\u0130ADA MAH\', \'pk\': \'34973\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ADALAR\', \'semt_bucak_belde\': \'KINALIADA\', \'mahalle\': \'KINALIADA MAH\', \'pk\': \'34977\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'ANADOLU MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'ARNAVUTK\\u00d6Y MERKEZ MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'\\u0130MRAHOR MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'\\u0130SLAMBEY MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'MUSTAFA KEMAL PA\\u015eA MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'NENEHATUN MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'ARNAVUTK\\u00d6Y\', \'mahalle\': \'YAVUZ SEL\\u0130M MAH\', \'pk\': \'34275\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'BAKLALI\', \'mahalle\': \'ADNAN MENDERES MAH\', \'pk\': \'34277\'}, {\'plaka\': 34, \'il\': \'\\u0130STANBUL\', \'ilce\': \'ARNAVUTK\\u00d6Y\', \'semt_bucak_belde\': \'BAKLALI\', \'mahalle\': \'BAKLALI MAH\', \'pk\': \'34277\'}, {\'plaka\': 34, "}']}]}, {'subtask': "Use the 'Transitaires' tool with API 2 to retrieve all transit agencies in Istanbul, Turkey.", 'tool': '916', 'answer': 'Based on the API response, there are several transit agencies located in Istanbul, Turkey. Here are some details about each agency:\n\n1. EKVF: This transit agency has an ID of "EKVF" and the name "EKVF". Their phone number is "+687 27.25.92."\n\n2. ACT - Agence Calédonienne de Transit: With an ID of "ACT\\_AGENCE\\_CALEDONIENNE\\_DE\\_TRANSIT", this agency goes by the name "ACT - Agence Calédonienne de Transit". Their contact information includes a code postal of "98800", email address "sales@act.nc", and phone number "+687 27.55.48". Additionally, they have a ride ticket number of "0036590".\n\n3. AFL - Agence de Fréight et Logistique: This transit agency\'s ID is "AFL\\_AGENCE\\_DE\\_FRET\\_ET\\_LOGISTIQUE". They operate under the name "AFL - Agence de Fréight et Logistique". Their contact details consist of a code postal of "98846", email address "ltn@ltn.nc", phone number "+687 26.11.11", and website address "www.transportinter.eu".\n\n4. Agence Logistique et Transit Nouméa: Their ID is "AGENCE\\_LOGISTIQUE\\_ET\\_TRANSIT\\_NOUMEA". They go by the name "Agence Logistique et Transit Nouméa". Their contact information includes a code postal of "98800", email address "ltn@ltn.nc", phone number "+687 24.21.85", and website address "www.ltn.nc".\n\n5. ATI TRAMAR-Groupe Léon Vincent: This transit agency holds an ID of "ATI\\_TRAMAR\\_GROUPE\\_LEON\\_VINCENT" and operates under the name "ATI TRAMAR-Groupe Léon Vincent". Their contact information consists of a phone number "+687 28.61.25".\n\n6. ATL Lines DEMLINES: With an ID of "ATL\\_LINES\\_DEMLINES", this transit agency goes by the name "ATL Lines - DEMLINES". Their contact details include an email address "e.fleurence@atl-lines.fr" and website address "nouvellecaledonie.demlines.com".\n\nThese are all the transit agencies retrieved from the API response for Istanbul, Turkey.', 'current_final_answer': '', 'api_response': [{'name': 'transitaires', 'parameter': {}, 'response': ['{"error": "", "response": "[{\\"id\\":\\"EKVF\\",\\"name\\":\\"EKVF\\",\\"phone\\":\\"+687 27.25.92\\"},{\\"id\\":\\"ACT_AGENCE_CALEDONIENNE_DE_TRANSIT\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"sales@act.nc\\",\\"name\\":\\"ACT - Agence Cal\\u00e9donienne de Transit\\",\\"phone\\":\\"+687 27.55.48\\",\\"ridet\\":\\"0036590\\"},{\\"id\\":\\"AFL_AGENCE_DE_FRET_ET_LOGISTIQUE\\",\\"codePostal\\":\\"98846\\",\\"name\\":\\"AFL - Agence de Fr\\u00eat et Logistique\\",\\"phone\\":\\"+687 26.11.11\\",\\"website\\":\\"www.transportinter.eu\\"},{\\"id\\":\\"AGENCE_LOGISTIQUE_ET_TRANSIT_NOUMEA\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"ltn@ltn.nc\\",\\"name\\":\\"Agence Logistique et Transit Noum\\u00e9a\\",\\"phone\\":\\"+687 24.21.85\\",\\"website\\":\\"www.ltn.nc\\"},{\\"id\\":\\"ATI_TRAMAR_GROUPE_LEON_VINCENT\\",\\"name\\":\\"ATI TRAMAR- Groupe L\\u00e9on Vincent\\",\\"phone\\":\\"+687 28.61.25\\"},{\\"id\\":\\"ATL_LINES_DEMLINES\\",\\"email\\":\\"e.fleurence@atl-lines.fr\\",\\"name\\":\\"ATL Lines - DEMLINES\\",\\"website\\":\\"nouvellecaledonie.demlines.com\\"},{\\"id\\":\\"BOLLORE_LOGISTICS_NOUMEA\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"nc.commercial@bollore.com\\",\\"name\\":\\"BOLLORE LOGISTICS - Noum\\u00e9a\\",\\"phone\\":\\"+687 28.32.43\\",\\"ridet\\":\\"0590778\\"},{\\"id\\":\\"COMIMEX\\",\\"email\\":\\"comimex.nc@gmail.com\\",\\"name\\":\\"Comimex\\",\\"phone\\":\\"+687 28.43.41\\",\\"ridet\\":\\"0882944\\"},{\\"id\\":\\"COTRANS_CALEDONIE_TRANSIT\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"cotrans@cotrans.nc\\",\\"name\\":\\"COTRANS Cal\\u00e9donie Transit\\",\\"phone\\":\\"+687 27.50.52\\",\\"ridet\\":\\"0034033\\"},{\\"id\\":\\"CTL_CONSULTANT_TRANPSORT_LOGISTIQUE\\",\\"codePostal\\":\\"98800\\",\\"name\\":\\"CTL - Consultant Tranpsort Logistique\\",\\"phone\\":\\"+687 28.69.18\\"},{\\"id\\":\\"CALEDONIE_TRANSIT\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"cotrans@cotrans.nc\\",\\"name\\":\\"Cal\\u00e9donie Transit\\",\\"phone\\":\\"+687 27.50.52\\"},{\\"id\\":\\"DEM_PACIFIC\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"direction@dempacific.nc\\",\\"name\\":\\"DEM-pacific\\",\\"phone\\":\\"+687 26.21.85\\",\\"ridet\\":\\"0747634\\",\\"website\\":\\"dempacific.nc\\"},{\\"id\\":\\"DHL_GLOBAL_FORWARDING\\",\\"codePostal\\":\\"98845\\",\\"name\\":\\"DHL - Global forwarding\\",\\"phone\\":\\"+687 24.69.00\\",\\"ridet\\":\\"0383588\\",\\"website\\":\\"www.dhl.com\\"},{\\"id\\":\\"DOUANE_AGENCE_GONDRAND\\",\\"codePostal\\":\\"98800\\",\\"email\\":\\"direction@gondrand.nc\\",\\"name\\":\\"Douane Agence Gondrand\\",\\"phone\\":\\"+687 27.41.18\\"},{\\"id\\":\\"FEDEX_EXPRESS\\",\\"codePostal\\":\\"98800\\",\\"name\\":\\"FedEx Express\\",\\"phone\\":"}']}]}]

# # # answer_detail = get_answer_details(final_answer, previous_log)
# # # with open("/home/chenjunzhi/Modulized_Prompt_LLM/Inference/answer_detail.json", "w") as file:
# # #     json.dump(answer_detail, file, ensure_ascii=False, indent=4)

def test(query_file, tool_doc_dir, white_list, output_file, chat, start_index, whole_solution_file):
    with open(query_file) as file:
        query_json = json.load(file)
    with open(tool_doc_dir) as file:
        tool_doc = json.load(file)
    total_query = len(query_json)
    output_file_json = {}
    solution_file_json = {}
    if start_index != 0:
        with open(output_file, 'r') as file:
            output_file_json = json.load(file)
        with open(whole_solution_file, 'r') as file:
            solution_file_json = json.load(file)
    with tqdm(total=total_query, desc="Processing files", initial=start_index) as pbar:
        for i, test_query in enumerate(query_json[start_index:], start=start_index):
            # test_query = query_json[0]
            query = test_query["query"]
            relevant_APIs = test_query["relevant APIs"]
            api_list = test_query["api_list"]
            tool_dic = test_query["Tool_dic"]
            final_answer, previous_log, task_log,re_time, previous_log_totals = decompose_inference(query, relevant_APIs, api_list, white_list, tool_doc, tool_dic, chat)
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
            output_file_json[str(i)] = output_file_ele
            solution_file_ele = {
                "query": query,
                "total_steps": solution_total_steps,
                "task_log": task_log,
                "final_answer": final_answer,
                "answer_path": answer_details,
                "total_path": solution_tree
            }
            solution_file_json[str(i)] = solution_file_ele
            with open(output_file, "w") as file:
                json.dump(output_file_json, file, ensure_ascii=False, indent=4)
            with open(whole_solution_file, "w") as file:
                json.dump(solution_file_json, file, ensure_ascii=False, indent=4)
            pbar.update(1)

def run(query_file, tool_doc_dir, tool_env_dir, output_file, chat, whole_solution_file):
    with open(query_file, 'r') as file:
        input_file_json = json.load(file)
    total_len = len(input_file_json)
    white_list = get_white_list(tool_env_dir)
    while True:
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r') as file:
                    output_file_json = json.load(file)
                start_index = len(output_file_json)
                if start_index >= total_len:
                    # return
                    break
                test(query_file, tool_doc_dir, white_list, output_file, chat, start_index, whole_solution_file)
            else:
                test(query_file, tool_doc_dir, white_list, output_file, chat, 0, whole_solution_file)
        except Exception as e:
            print(e)
            print("some error occurs, continue...")
            time.sleep(60)
            continue



run("/home/chenjunzhi/JARVIS/easytool/data_toolbench/test_data/G2_category.json", "/home/chenjunzhi/JARVIS/easytool/data_toolbench/tool_instruction/toolbench_tool_instruction.json", "/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools", "data/mistral_no_easytool/decompose_answer_v5_g2.json", chat, "data/mistral_no_easytool/decompose_whole_v5_g2.json") 
run("/home/chenjunzhi/JARVIS/easytool/data_toolbench/test_data/G3_instruction.json", "/home/chenjunzhi/JARVIS/easytool/data_toolbench/tool_instruction/toolbench_tool_instruction.json", "/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools", "data/mistral_no_easytool/decompose_answer_v5_g3.json", chat, "data/mistral_no_easytool/decompose_whole_v5_g3.json") 


