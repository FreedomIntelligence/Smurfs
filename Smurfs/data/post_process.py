import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(sys.path)
import json
from Smurfs.data.utils import tree_steps_counter, total_path_transform
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="dir_to_your_data", required=False, help='the directory of the data that needs post-processing')
    parser.add_argument('--example_dir', type=str, default="dir_to_example_data", required=False, help='the directory of the example data')
    parser.add_argument('--test_sets', nargs='+', type=str, required=False, help='the test sets that need processing. It should be G2_instruction, G2_category or G3_instruction')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    input_dir = args.input_dir
    test_sets = args.test_sets
    example_dir = args.example_dir
    for test_set in test_sets:
        data_path = os.path.join(input_dir, f"{test_set}_raw.json")
        example_data_path = os.path.join(example_dir, f"{test_set}.json")

        with open(data_path, 'r') as file:
            g_data = json.load(file)
        with open(example_data_path, 'r') as file:
            g_example_data = json.load(file)
        g_new_data = {}
        for g_d in g_data:
            m = False
            for d in g_example_data:
                if g_data[g_d]["query"] == g_example_data[d]["query"]:
                    g_new_data_ele = {"query":"", "available_tools": [], "answer":{}}
                    g_new_answer_ele = {
                        "method": "smurfs",
                        "total_steps": 0,
                        "final_answer": "",
                        "answer_details": []
                        }
                    g_new_data_ele["query"] = g_data[g_d]["query"]
                    g_new_data_ele["available_tools"] = g_example_data[d]["available_tools"]
                    g_new_answer_ele["answer_details"] = [g_data[g_d]["answer"]["answer_details"]]
                    counter = tree_steps_counter(0)
                    counter.count_total_steps(g_data[g_d]["answer"]["answer_details"])
                    g_new_answer_ele["total_steps"] = counter.get_steps()
                    g_new_answer_ele["final_answer"] = g_data[g_d]["answer"]["final_answer"]
                    g_new_data_ele["answer"] = g_new_answer_ele
                    g_new_data[d] = g_new_data_ele
                    m = True
                    break
            if not m:
                print(f"{test_set} mismatch! The key is: {g_d}")
        if test_set == "G2_category":
            duplicate = g_new_data["43201"]
            g_new_data["43200"] = duplicate
        
        output_path = os.path.join(input_dir, f"{test_set}.json")
        print(output_path)
        with open(output_path, 'w') as file:
            json.dump(g_new_data, file, indent=4, ensure_ascii=False)
 
if __name__ == '__main__':
    main()