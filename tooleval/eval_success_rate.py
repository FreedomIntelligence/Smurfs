from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import os
import argparse
from tqdm import tqdm
from utils import test_sets
from langchain import LLMChain
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
import json
abs_dir = os.path.split(__file__)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--evaluator', type=str, default="tooleval_gpt-3.5-turbo_default", required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    return parser.parse_args()


def answer_check(question, answer, chat):
    while True:
        try:
            template = "You are a helpful assistant."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                "Please check whether the response can reasonably and accurately answer the question."
                "If can, please output 'YES'; If not, please output 'NO'\n"
                "You need to give reasons first and then decide whether the response can reasonably and accurately answer the question. You must only output in a parsible JSON format. Two example outputs look like:\n"
                "Example 1: {{\"Reason\": \"The reason why you think the response can reasonably and accurately answer the question\", \"Choice\": \"Yes\"}}\n"
                "Example 2: {{\"Reason\": \"The reason why you think the response cannot reasonably and accurately answer the question\", \"Choice\": \"No\"}}\n"
                "This is the user's question: {question}\n"
                "This is the response: {answer}\n"
                "Output: "
            )
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(question=question, answer=answer)
            # print(result)
            if 'yes'.lower() in str(result).lower():
                return 1, result
            else:
                return -1, result            
        except Exception as e:
            print(e)
            time.sleep(30)
            continue

    

if __name__ == '__main__':
    chat = ChatOpenAI(model_name="gpt-4", openai_api_key="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")
    args = parse_args()
    reference_model = args.reference_model
    output_list = []
    for test_set in test_sets:
        reference_path = f"{args.converted_answer_path}/{reference_model}/{test_set}.json"
        test_ids = list(json.load(open(os.path.join(args.test_ids, test_set+".json"), "r")).keys())
        reference_examples = json.load(open(reference_path, "r"))
        # for i in reference_examples:
        #     reference_examples[i] = remove_repeated_patterns(reference_examples[i], 4)
        if os.path.exists(f"{args.save_path}/{test_set}_{reference_model}.json"):
            existed_ids = list(json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r")).keys())
            label_cnt = json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r"))
        else:
            existed_ids = []
            label_cnt = {}
        
        for query_id in tqdm(reference_examples):
            if str(query_id) not in test_ids:
                continue
            if query_id in existed_ids:
                continue
            example = reference_examples[query_id]
            query = example["query"]
            answer = example["answer"]["final_answer"]
            success, result = answer_check(query, answer, chat)
            label_cnt[query_id] = {"query": "", "success": "", "result": {}, "final_answer": ""}
            label_cnt[query_id]["query"] = query
            label_cnt[query_id]["success"] = success
            label_cnt[query_id]["result"] = result
            label_cnt[query_id]["final_answer"] = answer
            json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)

        success_rate = 0
        for query_id in label_cnt:
            if label_cnt[query_id]["success"] == 1:
                success_rate += 1
        success_rate /= len(label_cnt)
        print(f"Test set: {test_set}. Model: {reference_model}. Success rate: {str(success_rate)}")        
# question = "My friends and I are eagerly awaiting the delivery of a package. Can you please track the package with the Pack & Send reference number 'ReferenceNumberHere'? Additionally, I'm interested in the latest status of the package with colis ID 'CA107308006SI'."
# answer = "Based on the information provided from the previous subtasks, I see that we have different outcomes for each part of your query. Regarding the first question about tracking the package with the Pack & Send reference number, unfortunately, neither the 'suivi-colis' nor 'Orderful' tools were successful in providing the desired tracking information due to errors encounter. In light of this issue, I suggested some troubleshooting steps for you to follow{\"Double check the reference number, contact Pack & Send customer support, and try again laterpie}.}\n\nHowever, good news came from the second question concerning the latest status of the package with colis ID CA107308006SI. The 'suivi-colis' tool reported that the package had already been delivered on September 9, 2019 at 9:41AM local timeNew Caledonia, Noumeaa. Therefore, your package has reached its final destination."

# result = answer_check(question, answer)
# print(result)
