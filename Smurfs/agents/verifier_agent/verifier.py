from Smurfs.agents.base import BaseAgent
from Smurfs.agents.verifier_agent.prompt import final_answer_check_prompt

class verifier_agent(BaseAgent):
    def __init__(self, llm, logger_dir):
        super().__init__(
            prompt = final_answer_check_prompt,
            llm = llm,
            name = "Verifier Agent",
            logger_dir = logger_dir
        )

    def run(self, question, answer, query_id):
        """agent run one step"""
        self.get_memory(question, answer)
        agent_prompt = self.get_prompt()
        message = [{'role': 'user', 
                 'content': agent_prompt}]
        ind = 0
        while True:
            try:
                result = self.llm.prediction(message)
                start = result.find("{")
                end = result.find("}")
                self.colorful_print(result, "Answer Verify")
                self.log(query_id, result)
                clean_result = eval(result[start:end+1])
                speak = clean_result["Speak"]
                status = clean_result["Status"]
                return speak, status
            except Exception as e:
                print(f"final answer check fails: {e}")
                self.log(query_id, f"final answer check fails: {e}")
                if ind > self.max_retry:
                    return -1, -1
                ind += 1
                continue

    def get_memory(self, question, answer):
        """get relevant memory and add it to agent's memory"""
        self.memory = {"question": question, "answer": answer}
    
    def get_prompt(self):
        """get the prompt for the agent"""
        agent_prompt = self.prompt.format(**self.memory)
        return agent_prompt