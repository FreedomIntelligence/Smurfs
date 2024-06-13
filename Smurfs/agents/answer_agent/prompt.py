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
This is the user's question: {question}
Output: """
tool_check_prompt = PromptTemplate.from_template(tool_check_prompt)

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

answer_generation_direct_prompt = """"You need to answer the user's question.
This is the user's question: {question}
Output:"""
answer_generation_direct_prompt = PromptTemplate.from_template(answer_generation_direct_prompt)

hotpot_answer_parser_prompt = """
You will need to extract a concise answer from the detailed answer to answer the question in a consice language style.
Only output your concise answer to the answer.
For example:
Question:
VIVA Media AG changed it's name in 2004. What does their new acronym stand for?
Detailed Answer:
The new name of VIVA Media AG since its change in 2004 is "VIVA Media GmbH". In this acronym, "GmbH" is a German term which means "Gesellschaft mit beschränkter Haftung", translating to "company with limited liability" in English. So, the acronym denotes that it is a type of business organization similar to a limited liability company (LLC). VIVA Media GmbH is a company that specializes in publishing, producing, and developing high-quality games for different platforms.
Output:
Gesellschaft mit beschränkter Haftung

Question:
Jaclyn Stapp is married to the former frontman of a band that disbanded in what year?
Detailed Answer:
The band Creed effectively ended on December 29, 2002. However, they had a reunion tour that started on August 6, 2009, and ended on October 20, 2009. They also released an album called "Full Circle" on October 27, 2009. Despite these reunions, the band's meteoric rise came to a halt when it split up again in 2004.
Output:
2004

Question:
{question}
Detailed Answer:
{detailed_answer}
Output:"""