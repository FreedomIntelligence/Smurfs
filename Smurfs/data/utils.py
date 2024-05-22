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

