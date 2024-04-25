from langchain.prompts import PromptTemplate

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