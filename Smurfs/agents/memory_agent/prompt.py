from langchain.prompts import PromptTemplate

mem_choose_prompt = """You are a memory agent that controls the memory of the agent system.
The agent system is trying to solve a complex question step by step by solving its subtasks one by one.
Among those subtasks, some subtask may need execution history of other subtasks to be solved.
Your task is to decide which subtasks' execution history is needed by the agent system to solve the current subtask.
Please note that:
1. If the current subtask is independent of the other subtasks, just output {{\"task\":}}
2.  
You must only output in a parsible JSON format. Two example outputs look like:
Example 1: {{\"Reason\": \"The reason why you think you do not need to call an external API to solve the user's question\", \"Choice\": \"No\"}}
Example 2: {{\"Reason\": \"The reason why you think you need to call an external API to solve the user's question\", \"Choice\": \"Yes\"}}
This is the current subtask: {question}
This is the previous execution history: {history}
Output: """
tool_check_prompt = PromptTemplate.from_template(mem_choose_prompt)