import json
import os

with open("data/mistral_answer/G2_category_raw.json", "r") as file:
    g2_data = json.load(file)

with open("data/mistral_answer/G2_instruction_raw.json", "r") as file:
    g2_data_ins = json.load(file)

with open("data/mistral_answer/G3_instruction_raw.json", "r") as file:
    g3_data = json.load(file)
#todo: 导入数据后需要重新写，最好写一个script来做
with open("/Users/chenjunzhi/Desktop/github_multi_agent/Multi_agents/data/mistral_multi/least_to_most_dfsdt_G2_v5_category.json", "r") as file:
    g2_gpt4_data = json.load(file)

with open("/Users/chenjunzhi/Desktop/github_multi_agent/Multi_agents/data/mistral_multi/least_to_most_dfsdt_G3_v5_instruction.json", "r") as file:
    g3_gpt4_data = json.load(file)

with open("/Users/chenjunzhi/Desktop/github_multi_agent/Multi_agents/data/mistral_multi/least_to_most_dfsdt_G3_v5_instruction.json", "r") as file:
    g3_gpt4_data = json.load(file)

class tree_steps_counter:
    def __init__(self, steps):
        self.steps = steps

    def count_total_steps(self, root):
        self.steps += 1
        if root["next"] == []:
            return
        for i in range(len(root["next"])):
            self.count_total_steps(root["next"][i])

    def get_steps(self):
        return self.steps

# counter = tree_steps_counter(0)
# counter.count_total_steps(root)
# steps = counter.get_steps()
# print(steps)
# c = 0 

def total_path_transform(data, index):
    finish_template = [{
                                "role": "tool",
                                "message": {
                                    "name": "Finish",
                                    "arguments": {
                                        "return_type": "give_answer",
                                        "final_answer": data[index]["final_answer"]
                                    },
                                    "response": ""
                                },
                                "next": []
                            }]
    answer_path = data[index]["total_path"]["next"][0]["next"]
    for i in range(len(answer_path)-1, -1, -1):
        if answer_path[i]["role"] ==  "plan_global":
            if answer_path[i]["next"] != []:
                current_log = answer_path[i]["next"][-1]
                answer_path[i] = answer_path[i]["next"]
            else:
                if i == len(answer_path)-1:
                    answer_path[i] = finish_template[0]
                continue
        else:
            current_log = answer_path[i]
        while current_log["next"] != []:
            current_log = current_log["next"][-1]
        if i == len(answer_path)-1:
            current_log["next"] = finish_template
        else:
            if not isinstance(answer_path[i+1], list):
                current_log["next"] = [answer_path[i+1]]
            else:
                current_log["next"] = answer_path[i+1]
    if not isinstance(answer_path[0], list):
        data[index]["total_path"]["next"][0]["next"] = [answer_path[0]]
    else:
        data[index]["total_path"]["next"][0]["next"] = answer_path[0]

g2_new_data = {}
for g2_d in g2_data:
    # total_path_transform(g2_data, g2_d)
        
            


    # print(g2_d)
    m = False
    for d in g2_gpt4_data:
        if g2_data[g2_d]["query"] == g2_gpt4_data[d]["query"]:
            g2_new_data_ele = {"query":"", "available_tools": [], "answer":{}}
            g2_new_answer_ele = {
                "method": "least_to_most_dfsdt",
                "total_steps": 0,
                "final_answer": "",
                "answer_details": []
                }
            g2_new_data_ele["query"] = g2_data[g2_d]["query"]
            g2_new_data_ele["available_tools"] = g2_gpt4_data[d]["available_tools"]
            g2_new_answer_ele["answer_details"] = [g2_data[g2_d]["answer"]["answer_details"]]
            counter = tree_steps_counter(0)
            counter.count_total_steps(g2_data[g2_d]["answer"]["answer_details"])
            g2_new_answer_ele["total_steps"] = counter.get_steps()
            g2_new_answer_ele["final_answer"] = g2_data[g2_d]["answer"]["final_answer"]
            g2_new_data_ele["answer"] = g2_new_answer_ele
            g2_new_data[d] = g2_new_data_ele
            m = True
            # c += 1
            break
    if not m:
        print(f"g2data mismatch! The key is: {g2_d}")
duplicate = g2_new_data["43201"]
g2_new_data["43200"] = duplicate
# print(c)

g2_new_data_ins = {}
for g2_d in g2_data_ins:
    # total_path_transform(g2_data, g2_d)
    # print(g2_d)
    m = False
    for d in g2_gpt4_data:
        if g2_data[g2_d]["query"] == g2_gpt4_data[d]["query"]:
            g2_new_data_ele = {"query":"", "available_tools": [], "answer":{}}
            g2_new_answer_ele = {
                "method": "least_to_most_dfsdt",
                "total_steps": 0,
                "final_answer": "",
                "answer_details": []
                }
            g2_new_data_ele["query"] = g2_data[g2_d]["query"]
            g2_new_data_ele["available_tools"] = g2_gpt4_data[d]["available_tools"]
            g2_new_answer_ele["answer_details"] = [g2_data[g2_d]["answer"]["answer_details"]]
            counter = tree_steps_counter(0)
            counter.count_total_steps(g2_data[g2_d]["answer"]["answer_details"])
            g2_new_answer_ele["total_steps"] = counter.get_steps()
            g2_new_answer_ele["final_answer"] = g2_data[g2_d]["answer"]["final_answer"]
            g2_new_data_ele["answer"] = g2_new_answer_ele
            g2_new_data_ins[d] = g2_new_data_ele
            m = True
            # c += 1
            break
    if not m:
        print(f"g2data mismatch! The key is: {g2_d}")
# duplicate = g2_new_data["43201"]
# g2_new_data["43200"] = duplicate

g3_new_data = {}
for g3_d in g3_data:
    # total_path_transform(g3_data, g3_d)
    m = False
    for d in g3_gpt4_data:
        if g3_data[g3_d]["query"] == g3_gpt4_data[d]["query"]:
            g3_new_data_ele = {"query":"", "available_tools": [], "answer":{}}
            g3_new_answer_ele = {
                "method": "least_to_most_dfsdt",
                "total_steps": 0,
                "final_answer": "",
                "answer_details": []
                }
            g3_new_data_ele["query"] = g3_data[g3_d]["query"]
            g3_new_data_ele["available_tools"] = g3_gpt4_data[d]["available_tools"]
            g3_new_answer_ele["answer_details"] = [g3_data[g3_d]["answer"]["answer_details"]]
            counter = tree_steps_counter(0)
            counter.count_total_steps(g3_data[g3_d]["answer"]["answer_details"])
            g3_new_answer_ele["total_steps"] = counter.get_steps()
            g3_new_answer_ele["final_answer"] = g3_data[g3_d]["answer"]["final_answer"]
            g3_new_data_ele["answer"] = g3_new_answer_ele
            g3_new_data[d] = g3_new_data_ele
            m = True
            break
    if not m:
        print(f"g3data mismatch! The key is: {g3_d}")

with open("data/mistral_answer/G2_category.json", "w") as file:
    json.dump(g2_new_data, file, indent=4)

with open("data/mistral_answer/G2_instruction.json", "w") as file:
    json.dump(g2_new_data_ins, file, indent=4)

with open("data/mistral_answer/G3_instruction.json", "w") as file:
    json.dump(g3_new_data, file, indent=4)

#必须要加入tool_finish!