import re
from collections import Counter
import string
import json

def format_step(step: str) -> str:
    step = step.strip('\n').strip().replace('\n', '')
    if step.startswith("Thought") or step.startswith("Action"):
        step = step.split()[2:]
        step = " ".join(step)
    if "Thought" in step:
        step = step.split("Thought")[0].strip()
    if "Action" in step:
        step = step.split("Action")[0].strip()
    if "Observation" in step:
        step = step.split("Observation")[0].strip()
    return step

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
  
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    elif " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0    

def eval_result_once(question, pre, gt):
    correct = EM(pre, gt)
    reward = f1_score(pre, gt)[0]
    # halted = agent.is_halted()
    # error = agent.run_error
    # prompt = agent._build_agent_prompt()
    save_dict = {"question":question, "answer":gt, "prediction": pre, "EM":correct, "reward":reward}
    # with open(file_path, 'a') as f:
    #     json.dump(save_dict, f)
    #     f.write("\n")
    return save_dict

def eval_result(eval_data):
    result = []
    parsed_result = []
    correct = 0
    reward = 0
    parsed_correct = 0
    parsed_reward = 0
    total_len = len(eval_data)
    for d in eval_data:
        pre = d["pre_ans"]
        parsed_pre = d["parsed_pre"]
        gt = d["gt_answer"]
        question = d["question"]
        pre_dict = eval_result_once(question, pre, gt)
        parsed_dict = eval_result_once(question, parsed_pre, gt)
        result.append(pre_dict)
        parsed_result.append(parsed_dict)

        correct += pre_dict["EM"]
        reward += pre_dict["reward"]

        parsed_correct += parsed_dict["EM"]
        parsed_reward += parsed_dict["reward"]

    correct /= total_len
    reward /= total_len

    parsed_correct /= total_len
    parsed_reward /= total_len

    return correct, reward, parsed_correct, parsed_reward, result, parsed_result



    