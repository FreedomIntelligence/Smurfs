from langchain.prompts import PromptTemplate

final_answer_check_prompt = """
An agent is trying to solve the query proposed by the user. \
You need to evaluate whether the given query has been completed reasonably and accurately. If so, summarize the solution to the user. If not, summarize the current progress, and propose what is missing.

You response contains following elements:
Speak: (your words to the agent if the task isn't completed, or a complete answer based on the full execution log to the user if the task is finished)
Status: (0 or 1. 0 for unfinished and 1 for finished)

Please note that: 
1. If the answer says the query can't be solved or it can't answer the query given the current information, please output Status as 0.
2. Only output Status as 1 if the query has been answered correctly and accurately.
3. If the answer only give a plan instead of a detailed answer, output Status as 0.

You must only output in a parsible JSON format. Two example outputs look like:
Example 1: {{\"Speak\": \"answer based on the full execution log to the user\", \"Status\": \"1\"}}
Example 2: {{\"Speak\": \"your words to the group if the task isn't solved\", \"Status\": \"0\"}}

This is the answer from the previous execution result:
{answer}

This is the original question: {question}
Output: """ 
final_answer_check_prompt = PromptTemplate.from_template(final_answer_check_prompt)