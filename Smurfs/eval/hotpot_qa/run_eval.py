import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Smurfs.inference.smurfs_worker import smurfs_worker
from Smurfs.tools.tool_env import tool_env
from Smurfs.model.openai_model.openai_model import OpenAI_Model
from Smurfs.agents.answer_agent.answer import answer_agent
from Smurfs.agents.executor_agent.executor import executor_agent
from Smurfs.agents.planning_agent.planner import hotpot_planning_agent
from Smurfs.agents.verifier_agent.verifier import verifier_agent
from Smurfs.eval.hotpot_qa.utils import eval_result
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

def run_one_hotpot(ques, ans, HP_answer_agent, worker, query_id):
    global results
    # ques, ans = task_instructions[0]
    # print(ques)
    # print(ans)
    pre = run(worker, ques, query_id)
    print(pre)

    # question = "Where was the first governor after the The Missouri Compromise from?"
    # detailed_answer = "The first governor of Missouri after the Missouri Compromise was Alexander McNair. He was originally from central Pennsylvania, specifically, he was born in Cumberland County on May 5, 1775, and later lived in Derry township, Lancaster (now Dauphin) County. He also pursued his education in Derry and attended the University of Pennsylvania in Philadelphia for a term."
    parsed_pre = HP_answer_agent.run(query_id=query_id, task="parse", question=ques, detailed_answer=pre)
    print(parsed_pre)

    result_ele = {"question": ques, "gt_answer": ans, "pre_ans": pre, "parsed_pre": parsed_pre, "id": query_id}
    lock.acquire()
    results.append(result_ele)
    lock.release()

def run_hotpot(query_list, HP_answer_agent, worker):
    with tqdm(total=len(query_list), desc="Processing files", initial=0) as pbar:
        for i, test_task_ins in enumerate(query_list, start=0):
            idx = test_task_ins[0]
            ques, ans = test_task_ins[1]
            while True:
                try:
                    run_one_hotpot(ques, ans, HP_answer_agent, worker, idx)
                    break
                except Exception as e:
                    print(e)
                    print("some error occurs, continue...")
                    time.sleep(60)
                    continue

            pbar.update(1)
        return

#测试三个测试集
# if __name__ == '__main__':
#     #store true pre and parse pre in a same json, calculate together
#     #dump them together
#     lock = threading.Lock()
#     levels = ['easy', 'medium', 'hard']
#     model_name = "gpt-3.5-turbo"
#     method_name = "GPT3-turbo-Smurfs"
#     llm = OpenAI_Model(model_name=model_name)
#     parser_llm = OpenAI_Model(model_name="gpt-4")
#     task_path = "/Users/chenjunzhi/Desktop/smurfs_more/AutoAct/Self_Plan/Group_Planning/benchmark_run/data/hotpotqa"
#     with open("/Users/chenjunzhi/Desktop/smurfs_more/Smurfs/Smurfs/tools/hotpot.json", "r") as f:
#         available_tools = json.load(f)
#     for level in levels:
#         # level = 'hard'
#         # model_name = "gpt-4-0613"
#         results = []
#         test_set = f"hotpot_qa_{level}"
        
#         output_dir = f"data/{method_name}/{test_set}/answer"
#         results_dir = f"data/{method_name}/{test_set}/results.json"
#         if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
#             os.makedirs(f"data/{method_name}/{test_set}/parser_log")
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         if os.path.exists(results_dir):
#             with open(results_dir, "r") as file:
#                 results = json.load(file)

#         items = os.listdir(output_dir)
#         for i in range(len(items)):
#             items[i] = items[i].split(".")[0]

#         HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
#         worker = smurfs_worker(available_tools, tool_env, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
#         hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
#         task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]


#         query_to_do = []
#         # if len(items) != 0:
#         for idx, q in enumerate(task_instructions):
#             # print(idx)
#             if str(idx) in items:
#                 continue
#             # query_id = q["query_id"]
#             # if str(query_id) not in test_ids:
#             #     continue
#             query_to_do_ele = (idx, q)
#             query_to_do.append(query_to_do_ele)

#         total_len = len(query_to_do)
#         query_len = len(task_instructions)
#         print(total_len)

#         threads = []
#         if total_len < 20:
#             for i in range(total_len):
#                 if total_len == 0:
#                     break
            
#                 start = i
#                 end = i+1
#                 if i == total_len-1:
#                     query_cur = query_to_do[start:]
#                 else:
#                     query_cur = query_to_do[start: end]
#                 t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
#                 t.start()
#                 threads.append(t)

#         else:
#             for i in range(20):        
                
#                 if total_len == 0:
#                     break
            
#                 start = round(total_len/20)*i
#                 end = round(total_len/20)*(i+1)
#                 if i == 19:
#                     query_cur = query_to_do[start:]
#                 else:
#                     query_cur = query_to_do[start: end]
#                 t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
#                 t.start()
#                 threads.append(t)
        
#         for thread in threads:
#             thread.join()

#         with open(results_dir, "w") as file:
#             json.dump(results, file, indent=4, ensure_ascii=False)
        
#         with open(results_dir, "r") as file:
#             eval_data = json.load(file)
        
#         correct, reward, parsed_correct, parsed_reward, pre_dict, parsed_dict = eval_result(eval_data)
#         print(f"correct rate for {test_set} is: {correct}, reward rate for {test_set} is: {reward}")
#         print(f"parsed correct rate for {test_set} is: {parsed_correct}, parsed reward rate for {test_set} is: {parsed_reward}")
        
#         with open(f"data/{method_name}/{test_set}/parsed_result.json", "w") as file:
#             json.dump(parsed_dict, file, indent=4, ensure_ascii=False)
        
#         with open(f"data/{method_name}/{test_set}/original_result.json", "w") as file:
#             json.dump(pre_dict, file, indent=4, ensure_ascii=False)







# 测试一个query
# if __name__ == '__main__':
#     #store true pre and parse pre in a same json, calculate together
#     #dump them together
#     lock = threading.Lock()
#     levels = ['easy', 'medium', 'hard']
#     model_name = "gpt-3.5-turbo"
#     method_name = "GPT3-test-Smurfs"
#     llm = OpenAI_Model(model_name=model_name)
#     parser_llm = OpenAI_Model(model_name="gpt-4")
#     task_path = "/Users/chenjunzhi/Desktop/smurfs_more/AutoAct/Self_Plan/Group_Planning/benchmark_run/data/hotpotqa"
#     with open("/Users/chenjunzhi/Desktop/smurfs_more/Smurfs/Smurfs/tools/hotpot.json", "r") as f:
#         available_tools = json.load(f)
#     # for level in levels:
#     level = 'hard'
#     # model_name = "gpt-4-0613"
#     results = []
#     test_set = f"hotpot_qa_{level}"
    
#     output_dir = f"data/{method_name}/{test_set}/answer"
#     results_dir = f"data/{method_name}/{test_set}/results.json"
#     if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
#         os.makedirs(f"data/{method_name}/{test_set}/parser_log")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if os.path.exists(results_dir):
#         with open(results_dir, "r") as file:
#             results = json.load(file)

#     items = os.listdir(output_dir)
#     for i in range(len(items)):
#         items[i] = items[i].split(".")[0]

#     HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
#     worker = smurfs_worker(available_tools, tool_env, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
#     hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
#     task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
#     ques, ans = task_instructions[15][0], task_instructions[15][1]
#     run_one_hotpot(ques, ans, HP_answer_agent, worker, 0)


#     # query_to_do = []
#     # if len(items) != 0:
#     # for idx, q in enumerate(task_instructions):
#     #     # print(idx)
#     #     if str(idx) in items:
#     #         continue
#     #     # query_id = q["query_id"]
#     #     # if str(query_id) not in test_ids:
#     #     #     continue
#     #     query_to_do_ele = (idx, q)
#     #     query_to_do.append(query_to_do_ele)

#     # total_len = len(query_to_do)
#     # query_len = len(task_instructions)
#     # print(total_len)

#     # threads = []
#     # if total_len < 20:
#     #     for i in range(total_len):
#     #         if total_len == 0:
#     #             break
        
#     #         start = i
#     #         end = i+1
#     #         if i == total_len-1:
#     #             query_cur = query_to_do[start:]
#     #         else:
#     #             query_cur = query_to_do[start: end]
#     #         t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
#     #         t.start()
#     #         threads.append(t)

#     # else:
#     #     for i in range(20):        
            
#     #         if total_len == 0:
#     #             break
        
#     #         start = round(total_len/20)*i
#     #         end = round(total_len/20)*(i+1)
#     #         if i == 19:
#     #             query_cur = query_to_do[start:]
#     #         else:
#     #             query_cur = query_to_do[start: end]
#     #         t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
#     #         t.start()
#     #         threads.append(t)
    
#     # for thread in threads:
#     #     thread.join()

#     with open(results_dir, "w") as file:
#         json.dump(results, file, indent=4, ensure_ascii=False)
    
#     # with open(results_dir, "r") as file:
#     #     eval_data = json.load(file)
    
#     # correct, reward, parsed_correct, parsed_reward, pre_dict, parsed_dict = eval_result(eval_data)
#     # print(f"correct rate for {test_set} is: {correct}, reward rate for {test_set} is: {reward}")
#     # print(f"parsed correct rate for {test_set} is: {parsed_correct}, parsed reward rate for {test_set} is: {parsed_reward}")
    
#     # with open(f"data/{method_name}/{test_set}/parsed_result.json", "w") as file:
#     #     json.dump(parsed_dict, file, indent=4, ensure_ascii=False)
    
#     # with open(f"data/{method_name}/{test_set}/original_result.json", "w") as file:
#     #     json.dump(pre_dict, file, indent=4, ensure_ascii=False)





#测试一个个测试集
if __name__ == '__main__':
    #store true pre and parse pre in a same json, calculate together
    #dump them together
    lock = threading.Lock()
    levels = ['easy', 'medium', 'hard']
    model_name = "gpt-3.5-turbo"
    method_name = "GPT3-turbo-Smurfs—2"
    llm = OpenAI_Model(model_name=model_name)
    parser_llm = OpenAI_Model(model_name="gpt-4")
    task_path = "/Users/chenjunzhi/Desktop/smurfs_more/AutoAct/Self_Plan/Group_Planning/benchmark_run/data/hotpotqa"
    with open("/Users/chenjunzhi/Desktop/smurfs_more/Smurfs/Smurfs/tools/hotpot.json", "r") as f:
        available_tools = json.load(f)
    # for level in levels:
    level = 'medium'
    # model_name = "gpt-4-0613"
    results = []
    test_set = f"hotpot_qa_{level}"
    
    output_dir = f"data/{method_name}/{test_set}/answer"
    results_dir = f"data/{method_name}/{test_set}/results.json"
    if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
        os.makedirs(f"data/{method_name}/{test_set}/parser_log")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(results_dir):
        with open(results_dir, "r") as file:
            results = json.load(file)

    items = os.listdir(output_dir)
    for i in range(len(items)):
        items[i] = items[i].split(".")[0]

    HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
    worker = smurfs_worker(available_tools, tool_env, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
    hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]


    query_to_do = []
    # if len(items) != 0:
    for idx, q in enumerate(task_instructions):
        # print(idx)
        if str(idx) in items:
            continue
        # query_id = q["query_id"]
        # if str(query_id) not in test_ids:
        #     continue
        query_to_do_ele = (idx, q)
        query_to_do.append(query_to_do_ele)

    total_len = len(query_to_do)
    query_len = len(task_instructions)
    print(total_len)

    threads = []
    if total_len < 20:
        for i in range(total_len):
            if total_len == 0:
                break
        
            start = i
            end = i+1
            if i == total_len-1:
                query_cur = query_to_do[start:]
            else:
                query_cur = query_to_do[start: end]
            t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
            t.start()
            threads.append(t)

    else:
        for i in range(20):        
            
            if total_len == 0:
                break
        
            start = round(total_len/20)*i
            end = round(total_len/20)*(i+1)
            if i == 19:
                query_cur = query_to_do[start:]
            else:
                query_cur = query_to_do[start: end]
            t = threading.Thread(target=run_hotpot, args=(query_cur, HP_answer_agent, worker))
            t.start()
            threads.append(t)
    
    for thread in threads:
        thread.join()

    with open(results_dir, "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    
    with open(results_dir, "r") as file:
        eval_data = json.load(file)
    
    correct, reward, parsed_correct, parsed_reward, pre_dict, parsed_dict = eval_result(eval_data)
    print(f"correct rate for {test_set} is: {correct}, reward rate for {test_set} is: {reward}")
    print(f"parsed correct rate for {test_set} is: {parsed_correct}, parsed reward rate for {test_set} is: {parsed_reward}")
    
    with open(f"data/{method_name}/{test_set}/parsed_result.json", "w") as file:
        json.dump(parsed_dict, file, indent=4, ensure_ascii=False)
    
    with open(f"data/{method_name}/{test_set}/original_result.json", "w") as file:
        json.dump(pre_dict, file, indent=4, ensure_ascii=False)
