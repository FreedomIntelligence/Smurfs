import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Smurfs.inference.smurfs_worker import smurfs_hotpot_worker, smurfs_worker
# from Smurfs.tools.tool_env import HotpotToolEnv
from Smurfs.tools.tool_env import tool_env
from Smurfs.model.openai_model.openai_model import OpenAI_Model, OpenRouter_Model
from Smurfs.agents.answer_agent.answer import answer_agent
from Smurfs.agents.executor_agent.executor import executor_agent
from Smurfs.agents.planning_agent.planner import hotpot_planning_agent
from Smurfs.agents.verifier_agent.verifier import verifier_agent
import json
import threading
import joblib
from tqdm import tqdm
import time

def run(worker, query, query_id):
    # global lock
    final_answer, output_file_ele, solution_file_ele = worker.run(query, query_id)
    # lock.acquire()
    worker.save_solution(output_file_ele, solution_file_ele, query_id)
    # lock.release()
    return final_answer

def cli_run(query, worker):
    pre = run(worker, query, 0)
    return pre

if __name__ == '__main__':
    lock = threading.Lock()
    # model_name = "mistralai/mistral-7b-instruct-v0.2"
    model_name = "gpt-4"
    method_name = "cli_inference"
    tool_doc_path = "Smurfs/tools/math_search.json"
    llm = OpenAI_Model(model_name=model_name)
    # llm = OpenRouter_Model(model_name=model_name)
    # parser_llm = OpenAI_Model(model_name="gpt-4")
    with open(tool_doc_path, "r") as f:
        available_tools = json.load(f)

    test_set = "cli"
        
    output_dir = f"data/{method_name}/{test_set}/answer"
    results_dir = f"data/{method_name}/{test_set}/results.json"
    if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
        os.makedirs(f"data/{method_name}/{test_set}/parser_log")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
    # worker = smurfs_hotpot_worker(available_tools, HotpotToolEnv, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
    worker = smurfs_worker(available_tools, tool_env, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
    query = input("Please Enter Your Task: ")
    cli_run(query, worker)
