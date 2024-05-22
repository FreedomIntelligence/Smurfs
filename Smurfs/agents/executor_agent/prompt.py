from langchain.prompts import PromptTemplate

generate_thought_prompt = """\
You need to analyse the previous execution history and generate your internal reasoning and thoughts on the task, and how you plan to solve it based on the current attempts.

Do not output thought that is too long. Output in 2-3 sentences is OK.

This is the user's task: 
{question}

This is the Tool List: 
{tool_list}

This is the previous execution history:
{previous_log}

This is the hint comes from the evaluator:
{hint}

Output:"""
generate_thought_prompt = PromptTemplate.from_template(generate_thought_prompt)

choose_tool_prompt = """\
This is the user's question: 
{question}
These are the tools you can select to solve the question:
Tool List:
{tool_list}

Please note that: 
1. You should only chooce one tool from the Tool List to solve this question.
2. You must ONLY output the ID of the tool and your reason for choosing it in a parsible JSON format. An example output looks like:
'''
Example: {{\"ID\": ID of the tool, \"Reason\": The reason for choosing the tool}}
'''
Output: """
choose_tool_prompt = PromptTemplate.from_template(choose_tool_prompt)

choose_parameter_prompt="""\
Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.
Please note that: 
1. The Example in the API tool documentation can help you better understand the use of the API.
2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{}}
3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.
4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.
5. You must ONLY output in a parsible JSON Format. The example output looks like:
'''
Example: {{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}
'''

There are logs of previous questions and answers: 
{previous_log}

This is the current user's question: {question}

This is API tool documentation: {api_dic}
Output:"""
choose_parameter_prompt = PromptTemplate.from_template(choose_parameter_prompt)