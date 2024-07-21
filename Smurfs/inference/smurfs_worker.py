# — coding: utf-8 –
import json
import sys
import argparse
import time
import requests
import os
from Smurfs.inference.utils import change_name, standardize, get_white_list, get_answer_log, get_observation_log, build_tree, get_answer_details, test_sets
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Smurfs.model.vllm_model.vllm_model import vllm_Model
from Smurfs.inference.server import get_rapidapi_response
import threading
from Smurfs.agents.answer_agent.answer import answer_agent
from Smurfs.agents.executor_agent.executor import executor_agent
from Smurfs.agents.planning_agent.planner import planning_agent
from Smurfs.agents.verifier_agent.verifier import verifier_agent
from termcolor import colored  
class smurfs_worker:
    def __init__(self, available_tools, tool_env, llm, method_name, test_set, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent):
        #available_tools的格式形如toolbench里面的api_list里的格式，只需要api_name
        #tool_env是一个工具函数里用来存储工具代码的py文件中的所有函数的字典，key为函数名，value是函数对象
        self.available_tools = available_tools
        
        self.output_dir = f"data/{method_name}/{test_set}/answer"

        self.whole_solution_dir = f"data/{method_name}/{test_set}/whole"

        self.logger_dir = f"data/{method_name}/{test_set}/agent_log"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.whole_solution_dir):
            os.makedirs(self.whole_solution_dir)
        
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

        self.Answer_Agent = Answer_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Executor_Agent = Executor_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Planning_Agent = Planning_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Verifier_Agent = Verifier_Agent(llm=llm, logger_dir=self.logger_dir)
        self.tool_env = tool_env
    
    def inference(self, query, relevant_APIs, subtask, query_id, max_step=3):
        # tool_check_num, reason = self.Answer_Agent.run(question=query, task="tool_check", query_id=query_id)
        # #direct answer
        # if tool_check_num == 1:
        #     # input_dic = {"task": query}
        #     answer = self.Answer_Agent.run(question=query, task="direct", query_id=query_id)
        #     previous_log = [{"thought": reason, "action": "", "action_input": "", "observation": "", "answer": answer, "tool": "","id": 0}]
        #     history_log = [{"thought": reason, "action": "", "action_input": "", "observation": "", "answer": answer, "previous_id": -1, "id": 0}]
        #     return answer, previous_log, 0, history_log

        previous_log = []
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
        retry_parameter = 0
        re_time = 0
        subtask_id = 0
        restart = 0
        while True:
            if step_num >= max_step:
                print("\n\nReach steps limits, return answers!\n\n")
                answer_log = get_answer_log(history_log)
                answer = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
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

            if len(tool_list) == 0:
                if len(previous_log) == 0:
                    answer_log = get_answer_log(history_log)
                    partial_answer = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
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
                thought = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")

            else:
                thought = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")
                tool_id = self.Executor_Agent.run(question=subtask, tool_list=tool_list, thought=thought, query_id=query_id, task="tool")
            
            try:
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
            
            # tool_name_list = tool_des_json["tool_name"].split(":")
            # category_name = tool_name_list[0]
            # tool_name = tool_name_list[1]
            api_name = tool_des_json["tool_name"]
            API_doc = tool_des_json
            
            while True:
                try:
                    parameters = {}

                    if retry_parameter == 4:
                        restart = 1
                        retry_parameter = 0
                        print("No Para! Restart!")
                        break
                    
                    parameter = self.Executor_Agent.run(api_dic=API_doc, question=query, previous_log=answer_log, thought=thought, query_id=query_id, task="parameter")
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
        
            
            # api_name = change_name(standardize(api_name))

            if restart != 1:
                try:
                    observation = self.Call_function(api_name, parameters)
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
            print("##########Tool Response##########")
            print(f"{observation}\n")
            observation_log = get_observation_log(previous_log)

            answer = self.Answer_Agent.run(question=subtask, call_result=observation_log, query_id=query_id, task="answer")

            previous_log[-1]["answer"] = answer

            history_log_ele = {"thought": thought, "action": api_name, "action_input": parameters, "observation": observation, "answer": answer, "previous_id": previous_id, "id": subtask_id}
            history_log.append(history_log_ele)
            subtask_id += 1

            speak, status = self.Verifier_Agent.run(question=subtask, answer=answer, query_id=query_id)
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

    def decompose_inference(self, query, query_id):
        while True:
            subtasks = self.Planning_Agent.run(question=query, query_id=query_id)
            if subtasks == -1:
                continue
            break
        task_log = ""
        history_log = []
        previous_log_totals = []
        re_time_total = 0
        # print(subtasks)
        relevant_API_list = {}
        tool_id = 0
        for api in self.available_tools:
            tool_name = api["api_name"]
            ele = {"ID": tool_id, "tool_name": tool_name, "description": api["api_description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"]}
            relevant_API_list[str(tool_id)] = ele
            tool_id += 1

        for subtask in subtasks:
            task_log += f"question: {subtask}\n"
            answer, previous_log, re_time, previous_log_total = self.inference(task_log, relevant_API_list, subtask, query_id)
            previous_log_totals.append(previous_log_total)
            # print(answer)
            history_log += previous_log
            re_time_total += re_time
            task_log += f"answer: {answer}\n"
        final_answer = self.Answer_Agent.run(question=query, previous_log=task_log, task="final", query_id=query_id)
        return final_answer, history_log, task_log, re_time_total, previous_log_totals 

    def run(self, input, query_id):
        # result = {}
        # st = time.time()
        final_answer, previous_log, task_log,re_time, previous_log_totals = self.decompose_inference(input, query_id)
        answer_details, total_steps = get_answer_details(final_answer, previous_log)
        solution_tree, solution_total_steps = build_tree(previous_log_totals, task_log)
        output_file_ele = {
            "query": input,
            "restart_time": re_time,
            "answer": {
                "method": "decompose_dfs",
                "total_steps": total_steps,
                "final_answer": final_answer,
                "answer_details": answer_details
            }
        }
                    
        solution_file_ele = {
            "query": input,
            "total_steps": solution_total_steps,
            "task_log": task_log,
            "final_answer": final_answer,
            "answer_path": answer_details,
            "total_path": solution_tree
        }
        return final_answer, output_file_ele, solution_file_ele

    def save_solution(self, output_file_ele, solution_file_ele, idx):
        file_name = f"{idx}.json"
        output_file = os.path.join(self.output_dir, file_name)
        whole_solution_file = os.path.join(self.whole_solution_dir, file_name)
        with open(output_file, "w") as file:
            json.dump(output_file_ele, file, ensure_ascii=False, indent=4)
        with open(whole_solution_file, "w") as file:
            json.dump(solution_file_ele, file, ensure_ascii=False, indent=4)

    def Call_function(self, tool_name, args):
        try:
            print(tool_name)
            func = self.tool_env[tool_name]
            observation = func(**args)
            return observation
        except Exception as e:
            print(e)
            print(f"Call function fails")
            with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "id": 0,
                    "parameters": args,
                    "tool": tool_name,
                    "wrong": str(e)
                }, ensure_ascii=False)
                f.write(line + '\n')
            return -1
    
class smurfs_hotpot_worker:
    def __init__(self, available_tools, tool_env, llm, method_name, test_set, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent):
        #available_tools的格式形如toolbench里面的api_list里的格式，只需要api_name
        #tool_env是一个工具函数里用来存储工具代码的py文件中的所有函数的字典，key为函数名，value是函数对象
        self.available_tools = available_tools
        
        self.output_dir = f"data/{method_name}/{test_set}/answer"

        self.whole_solution_dir = f"data/{method_name}/{test_set}/whole"

        self.logger_dir = f"data/{method_name}/{test_set}/agent_log"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.whole_solution_dir):
            os.makedirs(self.whole_solution_dir)
        
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

        self.Answer_Agent = Answer_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Executor_Agent = Executor_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Planning_Agent = Planning_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Verifier_Agent = Verifier_Agent(llm=llm, logger_dir=self.logger_dir)
        self.tool_class = tool_env
        self.tool_env = {}
    
    def inference(self, query, relevant_APIs, subtask, query_id, max_step=3):
        # tool_check_num = self.Answer_Agent.run(question=query, task="tool_check", query_id=query_id)
        # #direct answer
        # if tool_check_num == 1:
        #     input_dic = {"task": query}
        #     answer = self.Answer_Agent.run(input_dic)
        #     return answer, answer, None, None

        previous_log = []
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
        retry_parameter = 0
        re_time = 0
        subtask_id = 0
        restart = 0
        while True:
            if step_num >= max_step:
                print("\n\nReach steps limits, return answers!\n\n")
                answer_log = get_answer_log(history_log)
                answer = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
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

            if len(tool_list) == 0:
                if len(previous_log) == 0:
                    answer_log = get_answer_log(history_log)
                    partial_answer = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
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
                thought = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")

            else:
                thought = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")
                tool_id = self.Executor_Agent.run(question=subtask, tool_list=tool_list, thought=thought, query_id=query_id, task="tool")
            
            try:
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
            
            # tool_name_list = tool_des_json["tool_name"].split(":")
            # category_name = tool_name_list[0]
            # tool_name = tool_name_list[1]
            api_name = tool_des_json["tool_name"]
            API_doc = tool_des_json
            
            while True:
                try:
                    parameters = {}

                    if retry_parameter == 4:
                        restart = 1
                        retry_parameter = 0
                        print("No Para! Restart!")
                        break
                    
                    parameter = self.Executor_Agent.run(api_dic=API_doc, question=query, previous_log=answer_log, thought=thought, query_id=query_id, task="parameter")
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
        
            
            # api_name = change_name(standardize(api_name))

            if restart != 1:
                try:
                    observation = self.Call_function(api_name, parameters)
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
            print("##########Tool Response##########")
            print(f"{observation}\n")
            previous_log.append(current_log)

            observation_log = get_observation_log(previous_log)

            answer = self.Answer_Agent.run(question=subtask, call_result=observation_log, query_id=query_id, task="answer")

            previous_log[-1]["answer"] = answer

            history_log_ele = {"thought": thought, "action": api_name, "action_input": parameters, "observation": observation, "answer": answer, "previous_id": previous_id, "id": subtask_id}
            history_log.append(history_log_ele)
            subtask_id += 1

            speak, status = self.Verifier_Agent.run(question=subtask, answer=answer, query_id=query_id)
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

    def decompose_inference(self, query, query_id):
        while True:
            subtasks = self.Planning_Agent.run(question=query, query_id=query_id)
            if subtasks == -1:
                continue
            break
        task_log = ""
        history_log = []
        previous_log_totals = []
        re_time_total = 0
        # print(subtasks)
        relevant_API_list = {}
        tool_id = 0
        for api in self.available_tools:
            tool_name = api["api_name"]
            ele = {"ID": tool_id, "tool_name": tool_name, "description": api["api_description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"]}
            relevant_API_list[str(tool_id)] = ele
            tool_id += 1

        for subtask in subtasks:
            task_log += f"question: {subtask}\n"
            answer, previous_log, re_time, previous_log_total = self.inference(task_log, relevant_API_list, subtask, query_id)
            previous_log_totals.append(previous_log_total)
            # print(answer)
            history_log += previous_log
            re_time_total += re_time
            task_log += f"answer: {answer}\n"
        final_answer = self.Answer_Agent.run(question=query, previous_log=task_log, task="final", query_id=query_id)
        return final_answer, history_log, task_log, re_time_total, previous_log_totals 

    def run(self, input, query_id):
        # result = {}
        # st = time.time()
        HPEnv = self.tool_class()
        self.tool_env = {
            "BingSearch": HPEnv.BingSearch,
            "Retrieve": HPEnv.Retrieve,
            "Lookup": HPEnv.Lookup
            }
        final_answer, previous_log, task_log,re_time, previous_log_totals = self.decompose_inference(input, query_id)
        answer_details, total_steps = get_answer_details(final_answer, previous_log)
        solution_tree, solution_total_steps = build_tree(previous_log_totals, task_log)
        output_file_ele = {
            "query": input,
            "restart_time": re_time,
            "answer": {
                "method": "decompose_dfs",
                "total_steps": total_steps,
                "final_answer": final_answer,
                "answer_details": answer_details
            }
        }
                    
        solution_file_ele = {
            "query": input,
            "total_steps": solution_total_steps,
            "task_log": task_log,
            "final_answer": final_answer,
            "answer_path": answer_details,
            "total_path": solution_tree
        }
        return final_answer, output_file_ele, solution_file_ele

    def save_solution(self, output_file_ele, solution_file_ele, idx):
        file_name = f"{idx}.json"
        output_file = os.path.join(self.output_dir, file_name)
        whole_solution_file = os.path.join(self.whole_solution_dir, file_name)
        with open(output_file, "w") as file:
            json.dump(output_file_ele, file, ensure_ascii=False, indent=4)
        with open(whole_solution_file, "w") as file:
            json.dump(solution_file_ele, file, ensure_ascii=False, indent=4)

    def Call_function(self, tool_name, args):
        try:
            print(tool_name)
            func = self.tool_env[tool_name]
            observation = func(**args)
            return observation
        except Exception as e:
            print(e)
            print(f"Call function fails")
            with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "id": 0,
                    "parameters": args,
                    "tool": tool_name,
                    "wrong": str(e)
                }, ensure_ascii=False)
                f.write(line + '\n')
            return -1


class stream_smurfs_worker:
    def __init__(self, available_tools, tool_env, llm, method_name, test_set, Answer_Agent, Executor_Agent, Planning_Agent, Verifier_Agent, OPENAI_API_KEY, BING_SUBSCRIPT_KEY, WOLFRAMALPH_APP_ID, WEATHER_API_KEYS):
        #available_tools的格式形如toolbench里面的api_list里的格式，只需要api_name
        #tool_env是一个工具函数里用来存储工具代码的py文件中的所有函数的字典，key为函数名，value是函数对象
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.BING_SUBSCRIPT_KEY = BING_SUBSCRIPT_KEY
        self.WOLFRAMALPH_APP_ID = WOLFRAMALPH_APP_ID
        self.WEATHER_API_KEYS = WEATHER_API_KEYS
        #print(self.BING_SUBSCRIPT_KEY)
        self.available_tools = available_tools
        
        self.output_dir = f"data/{method_name}/{test_set}/answer"

        self.whole_solution_dir = f"data/{method_name}/{test_set}/whole"

        self.logger_dir = f"data/{method_name}/{test_set}/agent_log"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.whole_solution_dir):
            os.makedirs(self.whole_solution_dir)
        
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

        self.Answer_Agent = Answer_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Executor_Agent = Executor_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Planning_Agent = Planning_Agent(llm=llm, logger_dir=self.logger_dir)
        self.Verifier_Agent = Verifier_Agent(llm=llm, logger_dir=self.logger_dir)
        self.tool_env = tool_env
    
    # def colorful_html(self, task, content, name):
    #     """print out message in different color"""
    #     role_to_color = {
    #     "Answer Agent": "red",
    #     "Executor Agent": "green",
    #     "Planning Agent": "blue",
    #     "Verifier Agent": "yellow",
    #     }
    #     color = role_to_color[name]
    #     html_text = f"<span style='color: {color}'>##########{task}##########<br>{content}<br></span>"
    #     # print(colored(f"##########{task}##########\n{content}\n", role_to_color[name]))
    #     return html_text

    def colorful_html(self, task, content, name):
        """print out message in different color"""
        role_to_color = {
        "Answer Agent": "red",
        "Executor Agent": "green",
        "Planning Agent": "blue",
        "Verifier Agent": "yellow",
        }
        color = role_to_color[name]
        if task != "Final Answer Generation":
            html_text = f"""<details><summary>{task}<br></summary>{content}<br></details>"""
        else:
            html_text = content
        # html_text = f"<span style='color: {color}'>##########{task}##########<br>{content}<br></span>"
        # print(colored(f"##########{task}##########\n{content}\n", role_to_color[name]))
        return html_text

    def inference(self, query, relevant_APIs, subtask, query_id, max_step=3):
        # tool_check_num, reason = self.Answer_Agent.run(question=query, task="tool_check", query_id=query_id)
        # #direct answer
        # if tool_check_num == 1:
        #     # input_dic = {"task": query}
        #     answer = self.Answer_Agent.run(question=query, task="direct", query_id=query_id)
        #     previous_log = [{"thought": reason, "action": "", "action_input": "", "observation": "", "answer": answer, "tool": "","id": 0}]
        #     history_log = [{"thought": reason, "action": "", "action_input": "", "observation": "", "answer": answer, "previous_id": -1, "id": 0}]
        #     return answer, previous_log, 0, history_log

        previous_log = []
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
        retry_parameter = 0
        re_time = 0
        subtask_id = 0
        restart = 0
        while True:
            if step_num >= max_step:
                yield("<br><br>Reach steps limits, return answers!<br><br>")
                answer_log = get_answer_log(history_log)
                answer, task, agent_name, result = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
                yield self.colorful_html(task, result, agent_name)
                yield answer, previous_log, re_time, history_log
                yield "stop"
            
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

            if len(tool_list) == 0:
                if len(previous_log) == 0:
                    answer_log = get_answer_log(history_log)
                    partial_answer, task, agent_name, result = self.Answer_Agent.run(question=query, previous_log=answer_log, task="final", query_id=query_id)
                    answer = f"Sorry, I can't answer this question accurately using the existing tools. A partial answer is: {partial_answer}"
                    yield self.colorful_html(task, result, agent_name)
                    yield answer, previous_log, re_time, history_log
                    yield "stop"
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
                thought, task, agent_name, result = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")
                yield self.colorful_html(task, result, agent_name)
            else:
                thought, task, agent_name, result = self.Executor_Agent.run(question=query, tool_list=tool_list, previous_log=answer_log, hint=hint, query_id=query_id, task="thought")
                yield self.colorful_html(task, result, agent_name)
                tool_id, task, agent_name, result = self.Executor_Agent.run(question=subtask, tool_list=tool_list, thought=thought, query_id=query_id, task="tool")
                yield self.colorful_html(task, result, agent_name)
            
            try:
                tool_id = int(tool_id)
                tool_id = str(tool_id)
                if tool_id not in relevant_APIs_ids:
                    re_time += 1
                    retry_tool_id += 1
                    yield("Tool ID wrong! Generate tool_id that do not exist!<br>")
                    continue
                tool_des_json = relevant_APIs[str(tool_id)]
                retry_tool_id = 0
            except:
                retry_tool_id += 1
                yield("Tool ID wrong! Generate tool_id that do not exist!<br>")
                continue
            
            # tool_name_list = tool_des_json["tool_name"].split(":")
            # category_name = tool_name_list[0]
            # tool_name = tool_name_list[1]
            api_name = tool_des_json["tool_name"]
            API_doc = tool_des_json
            
            while True:
                try:
                    parameters = {}

                    if retry_parameter == 4:
                        restart = 1
                        retry_parameter = 0
                        yield("No Para! Restart!<br>")
                        break
                    
                    parameter, task, agent_name, result = self.Executor_Agent.run(api_dic=API_doc, question=query, previous_log=answer_log, thought=thought, query_id=query_id, task="parameter")
                    yield self.colorful_html(task, result, agent_name)

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
                    yield("parameter generation fails, try again!<br>")
                    re_time += 1
                    continue
        
            
            # api_name = change_name(standardize(api_name))

            if restart != 1:
                try:
                    observation = self.Call_function(api_name, parameters)
                except:
                    observation = -1
                    
                if observation == -1:
                    restart = 1
                    observation = str({"error": "", "response": "call API fails"})

            if restart == 1:
                tool_used_dic[step_num].append(str(tool_id))
                yield('****Try Again For This Step****<br>')
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
            yield(f"<details><summary>Tool Response</summary>{observation}<br></details>")
            # print(f"{observation}\n")
            observation_log = get_observation_log(previous_log)

            answer, task, agent_name, result = self.Answer_Agent.run(question=subtask, call_result=observation_log, query_id=query_id, task="answer")
            yield self.colorful_html(task, result, agent_name)
            previous_log[-1]["answer"] = answer

            history_log_ele = {"thought": thought, "action": api_name, "action_input": parameters, "observation": observation, "answer": answer, "previous_id": previous_id, "id": subtask_id}
            history_log.append(history_log_ele)
            subtask_id += 1

            speak, status, task, agent_name, result = self.Verifier_Agent.run(question=subtask, answer=answer, query_id=query_id)
            yield self.colorful_html(task, result, agent_name)

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
                yield answer, previous_log, re_time, history_log
                yield "stop"

    def run(self, query, query_id):
        output = ""
        while True:
            subtasks, task, agent_name, result = self.Planning_Agent.run(question=query, query_id=query_id)
            if subtasks == -1:
                continue
            break
        output += self.colorful_html(task, result, agent_name)
        yield output
        task_log = ""
        history_log = []
        previous_log_totals = []
        re_time_total = 0
        # print(subtasks)
        relevant_API_list = {}
        tool_id = 0
        for api in self.available_tools:
            tool_name = api["api_name"]
            ele = {"ID": tool_id, "tool_name": tool_name, "description": api["api_description"], "required_parameters": api["required_parameters"], "optional_parameters": api["optional_parameters"]}
            relevant_API_list[str(tool_id)] = ele
            tool_id += 1

        for subtask in subtasks:
            sub_output = ""
            output_ele = "<details><summary>subtask: {subtask}</summary>{sub_output}</details>"
            task_log += f"question: {subtask}\n"
            inference_generator = self.inference(task_log, relevant_API_list, subtask, query_id)
            # old = None
            while True:
                try:
                    result = next(inference_generator)
                    # if result == "stop":
                    #     break
                    if isinstance(result, str):
                        sub_output += result
                        sub_out = output_ele.format(subtask=subtask, sub_output=sub_output)
                        output_ins = output + sub_out
                        yield output_ins
                    else:
                        break
                except StopIteration:
                    break
            output += sub_out
            answer, previous_log, re_time, previous_log_total = result
            #answer, previous_log, re_time, previous_log_total = self.inference(task_log, relevant_API_list, subtask, query_id)
            previous_log_totals.append(previous_log_total)
            # print(answer)
            history_log += previous_log
            re_time_total += re_time
            task_log += f"answer: {answer}\n"
        final_answer, task, agent_name, result = self.Answer_Agent.run(question=query, previous_log=task_log, task="final", query_id=query_id)
        output += self.colorful_html(task, result, agent_name)
        yield output
        # return final_answer, history_log, task_log, re_time_total, previous_log_totals 

    # def run(self, input, query_id):
    #     # result = {}
    #     # st = time.time()
    #     final_answer, previous_log, task_log,re_time, previous_log_totals = self.decompose_inference(input, query_id)
    #     answer_details, total_steps = get_answer_details(final_answer, previous_log)
    #     solution_tree, solution_total_steps = build_tree(previous_log_totals, task_log)
    #     output_file_ele = {
    #         "query": input,
    #         "restart_time": re_time,
    #         "answer": {
    #             "method": "decompose_dfs",
    #             "total_steps": total_steps,
    #             "final_answer": final_answer,
    #             "answer_details": answer_details
    #         }
    #     }
                    
    #     solution_file_ele = {
    #         "query": input,
    #         "total_steps": solution_total_steps,
    #         "task_log": task_log,
    #         "final_answer": final_answer,
    #         "answer_path": answer_details,
    #         "total_path": solution_tree
    #     }
    #     return final_answer, output_file_ele, solution_file_ele

    def save_solution(self, output_file_ele, solution_file_ele, idx):
        file_name = f"{idx}.json"
        output_file = os.path.join(self.output_dir, file_name)
        whole_solution_file = os.path.join(self.whole_solution_dir, file_name)
        with open(output_file, "w") as file:
            json.dump(output_file_ele, file, ensure_ascii=False, indent=4)
        with open(whole_solution_file, "w") as file:
            json.dump(solution_file_ele, file, ensure_ascii=False, indent=4)

    def Call_function(self, tool_name, args):
        try:
            print(tool_name)
            # print(self.BING_SUBSCRIPT_KEY)
            if tool_name == "bing_search" or tool_name == "BingSearch":
                args["key"] = self.BING_SUBSCRIPT_KEY
            if tool_name == "forecast_weather" or tool_name == "get_weather_today":
                args["KEY"] = self.WEATHER_API_KEYS
            if tool_name == "getWolframAlphaResults":
                args["APPID"] = self.WOLFRAMALPH_APP_ID
            
            print(args)
            func = self.tool_env[tool_name]
            observation = func(**args)
            return observation
        except Exception as e:
            print(e)
            print(f"Call function fails")
            with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "id": 0,
                    "parameters": args,
                    "tool": tool_name,
                    "wrong": str(e)
                }, ensure_ascii=False)
                f.write(line + '\n')
            return -1
  

        