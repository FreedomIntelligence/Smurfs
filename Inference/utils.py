# — coding: utf-8 –
import json
import re
import os
from tqdm import tqdm

def get_white_list(tool_root_dir):
    # print(tool_root_dir)
    white_list_dir = os.path.join(tool_root_dir)
    white_list = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        if not os.path.isdir(os.path.join(white_list_dir,cate)):
            continue
        for file in os.listdir(os.path.join(white_list_dir,cate)):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            # print(standard_tool_name)
            with open(os.path.join(white_list_dir,cate,file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {"description": js_data["tool_description"], "standard_tool_name": standard_tool_name}
    return white_list

def build_index(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name not in index:
                index[dir_name] = []
            index[dir_name].append(root)
    return index


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

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

def build_tree(previous_log_totals, task_log):
    
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

test_sets = ["G2_instruction", "G2_category", "G3_instruction"]
