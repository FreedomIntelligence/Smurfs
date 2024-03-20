import json
import random
with open("/Users/chenjunzhi/Desktop/reproduction_data/tooleval/pass_rate_results/G3_instruction_mistral_norepeat_least_to_most_dfsdt.json", 'r') as file:
    label_cnt = json.load(file)
pass_rate = 0
for query_id in label_cnt:
    if label_cnt[query_id]["failed"] < label_cnt[query_id]["passed"]:
        pass_rate += 1
    elif label_cnt[query_id]["failed"] == label_cnt[query_id]["passed"]:
        if random.random() < 0.5:
            pass_rate += 1
pass_rate /= len(label_cnt)
print(f"Pass rate: {str(pass_rate)}")