from langchain.prompts import PromptTemplate

tool_check_prompt = """As a powerful language model, you're equipped to answer user's question with accumulated knowledge.
However, in some cases, you need to use external APIs to answer accurately.
Thus, you need to check whether the user's question requires you to call an external API to solve it.
Here are some tips to help you check: 
1. If the user's question requires real-time information, since your knowledge base isn't updated in real-time, any such question will demand an API call.
2. If you need to obtain information (e.g., ID, name, phone number, geographical location, rank, etc.), you need to call the database APIs if you are not sure.
3. If the question demand a database search or internet research to generate an answer, this is another situation where an API call is necessary.
If need, please output 'YES'; If not, please output 'NO'
You need to give reasons first and then decide whether to keep it or not. You must only output in a parsible JSON format. Two example outputs look like:
Example 1: {{\"Reason\": \"The reason why you think you do not need to call an external API to solve the user's question\", \"Choice\": \"No\"}}
Example 2: {{\"Reason\": \"The reason why you think you need to call an external API to solve the user's question\", \"Choice\": \"Yes\"}}
This is the user's question: {task}
Output: """

tool_check_prompt = PromptTemplate.from_template(tool_check_prompt)

answer_generation_direct_prompt = """"You need to answer the user's question.
This is the user's question: {task}
Output:"""

answer_generation_direct_prompt = PromptTemplate.from_template(answer_generation_direct_prompt)

generate_subtask_prompt = """\
You need to analyse the previous execution history and generate your internal reasoning and thoughts on the task, and how you plan to solve it based on the current attempts.

Do not output thought that is too long. Output in 2-3 sentences is OK.

This is the user's task: 
{task}

This is the Tool List: 
{functions}

This is the previous execution history:
{messages}

This is the hint comes from the evaluator:
{hint}

Output:"""
generate_subtask_prompt = PromptTemplate.from_template(generate_subtask_prompt)

choose_tool_prompt = """\
This is the user's question: 
{question}
These are the tools you can select to solve the question:
Tool List:
{Tool_list}

Please note that: 
1. You should only chooce one tool from the Tool List to solve this question.
2. You must ONLY output the ID of the tool you chose in a parsible JSON format. An example output looks like:
'''
Example: {{\"ID\": XX}}
'''
Output: """
choose_tool_prompt = PromptTemplate.from_template(choose_tool_prompt)

#API_instruction是tool_des
#API_list是tool_doc中的api的字典的列表[{}, {}...]
choose_API_prompt = """\
{API_instruction}
This is a tool instruction. 
A task can be executed using api from this tool. Given this task, you should choose API from the tool to execute this subtask.
This is the API list: {API_list}
Please note that: 
1. The API name you choose must in the tool instruction.
2. The API description can be found in the tool instruction.
3. The Example in the API_list can help you better understand the use of the API.
4. You must ONLY output in the following parsible JSON Format. ONLY output the name of api. The example output looks like:

```
{{\"name\": api name}}
```
5. Directly output according to the output format.

Question: 
{question}
Output:"""
choose_API_prompt = PromptTemplate.from_template(choose_API_prompt)


choose_tool_prompt = """\
This is the user's question: {question}
These are the tools you can select to solve the question:
Tool List:
{Tool_list}

Please note that: 
1. You should only chooce one tool from the Tool List to solve this question.
2. You must ONLY output the ID of the tool you chose in a parsible JSON format. An example output looks like:
'''
Example: {{\"ID\": XX}}
'''
Output: """
choose_tool_prompt = PromptTemplate.from_template(choose_tool_prompt)


choose_parameter_prompt="""\
Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.
Please note that: 
1. The Example in the API tool documentation can help you better understand the use of the API.
2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{\"Parameters\":{{}}}}
3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.
5. You must ONLY output in a parsible JSON Format. The example output looks like:
'''
Example: {{\"Parameters\": {{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}}}
'''

There are logs of previous questions and answers: 
{previous_log}

This is the current user's question: {question}

This is API tool documentation: {api_dic}
Output:"""

choose_parameter_prompt = PromptTemplate.from_template(choose_parameter_prompt)

answer_generation_prompt = """
You should answer the question based on the response output by the API tool.
Please note that:
1. Answer the question in natural language based on the API response reasonably and effectively.
2. The user cannot directly get API response, so you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.
3. Do not output answer that is too long. Output in 3-6 sentences is OK.

This is the user's question:
{question}
This is the API response:
{call_result}
Output:"""
answer_generation_prompt = PromptTemplate.from_template(answer_generation_prompt)

final_answer_check_prompt = """
An agent is trying to solve the query proposed by the user. \
You need to evaluate whether the given query has been completed. If so, summarize the solution to the user. If not, summarize the current progress, and propose what is missing.

You response contains following elements:
Speak: (your words to the agent if the task is pending, or a complete answer based on the full execution log to the user if the task is finished)
Status: (0 or 1. 0 for pending and 1 for finished)

You must only output in a parsible JSON format. Two example outputs look like:
Example 1: {{\"Speak\": \"answer based on the full execution log to the user\", \"Status\": \"1\"}}
Example 2: {{\"Speak\": \"your words to the group if the task is pending\", \"Status\": \"0\"}}

There are logs of previous execution results: 
{previous_log}

This is the original question: {question}
Output: """ 
final_answer_check_prompt = PromptTemplate.from_template(final_answer_check_prompt)

##先生成答案再check还是先check再生成答案。如果先check再生成挺好的就不用先生成答案。


final_answer_generation_prompt = """
You will be given a complex question and you need to solve it step by step by decomposing it to a series of subtasks that can be solved using a single tool(functions).
At this step, you need to analyse the previous subtasks and their execution result to generate the answer to the original question reasonably and accurately.
Please note that:
1. Answer the question in natural language based on the subtask results reasonably and effectively.
2. The user cannot directly get the subtask results, so you need to make full use of the subtask results and give the information in the response that can satisfy the user's question in as much detail as possible.
This is the user's question:
{question}
There are logs of previous subtasks and execution results: 
{previous_log}
Output:"""
final_answer_generation_prompt = PromptTemplate.from_template(final_answer_generation_prompt)

task_decompose_prompt = """
You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.
Please note that: 
1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using a single tool.
2. Each simple subtask should be expressed into natural language.
3. Each subtask should contain the necessary information from the original question and should be complete, explicit and self-consistent.
4. You must ONLY output in a parsible JSON format. An example output looks like:
'''
{{\"Tasks\": [\"Task 1\", \"Task 2\", ...]}}
'''

This is the user's question: I'm planning a trip to Turkey and need information about postal codes in Istanbul. Can you provide me with the postal code and district for Istanbul province with plate number 34? Additionally, I would like to know if there are any transit agencies available in Istanbul. Please fetch their names and contact numbers.
Output: {{\"Tasks\": [\"Find the postal codes and districts for plate number 34 in Istanbul.\", \"Search for transit agencies and their contact numbers in Istanbul.\"]}}

This is the user's question: I recently moved to a new address and I need to update my information. Can you retrieve my address details using the postal code 75094080? Additionally, I would like to know the companies that offer shipping services.
Output: {{\"Tasks\": [\"retrieve the address details using the postal code 75094080\", \"search for companies that offer shipping services to my address\"]}}

This is the user's question: {question}
Output:
"""
task_decompose_prompt = PromptTemplate.from_template(task_decompose_prompt)