import gradio as gr

import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Smurfs.inference.smurfs_worker import smurfs_hotpot_worker, smurfs_worker, stream_smurfs_worker
# from Smurfs.tools.tool_env import HotpotToolEnv
from Smurfs.tools.tool_env import tool_env
from Smurfs.model.openai_model.openai_model import OpenAI_Model, OpenRouter_Model
from Smurfs.agents.answer_agent.answer import stream_answer_agent
from Smurfs.agents.executor_agent.executor import stream_executor_agent
from Smurfs.agents.planning_agent.planner import stream_hotpot_planning_agent
from Smurfs.agents.verifier_agent.verifier import stream_verifier_agent
import json
import threading
import joblib
from tqdm import tqdm
import time

# user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
# inp = gr.Textbox(placeholder="Enter your task")
# css = """
# .btn {background-color: blue; color: white;}
# #bot {background-color: blue; color: white;}
# #e {display: inline-block; vertical-align: middle;}
# """

# def update(name):
#     return f"<span style='color: red'>Welcome to Gradio, {name}!</span>"

def update(query):
    # model_name = "mistralai/mistral-7b-instruct-v0.2"
    model_name = "gpt-4"
    method_name = "cli_inference"
    tool_doc_path = "Smurfs/tools/math_search.json"
    # llm = OpenAI_Model(model_name=model_name)
    llm = OpenRouter_Model(model_name=model_name)
    # parser_llm = OpenAI_Model(model_name="gpt-4")
    with open(tool_doc_path, "r") as f:
        available_tools = json.load(f)

    test_set = "cli"
        
    output_dir = f"data/{method_name}/{test_set}/answer"
    results_dir = f"data/{method_name}/{test_set}/results.json"
    if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
        os.makedirs(f"data/{method_name}/{test_set}/parser_log")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
    # worker = smurfs_hotpot_worker(available_tools, HotpotToolEnv, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
    worker = stream_smurfs_worker(available_tools, tool_env, llm, method_name, test_set, stream_answer_agent, stream_executor_agent, stream_hotpot_planning_agent, stream_verifier_agent)
    stream_generator = worker.run(query, 0)
    # messages = []
    while True:
        try:
            response = next(stream_generator)
            messages = [(query, response)]
            yield messages
        except StopIteration:
            break
    # query = input("Please Enter Your Task: ")
    # cli_run(query, worker)

def test(content):
    result = ""
    for i in range(5):
        time.sleep(5)
        result+=content
        yield result

def user(user_msg):
    return user_msg


inp = gr.Textbox(placeholder="Please input your task")
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Smurfs</h1>""")    
    with gr.Row():
        with gr.Column(scale=1):
            inp.render()
            gr.Examples(["Who is the brother of the 2022 NBA FMVP?", "Calc integral of sin(x)+2x^2+3x+1 from 0 to 1"], inp)
            _submit = gr.Button("Submit")
            stop = gr.Button("Stop")
            clear = gr.Button("Clear")
            # btn = gr.Button("Run", elem_id="bot", elem_classes="btn")
        with gr.Column(scale=1, elem_id="e"):
            # chatbox = gr.HTML()
            chatbox = gr.Chatbot(height=20)
        # btn.click(fn=update, inputs=inp, outputs=chatbox)
        click_event = _submit.click(user, inp, inp).then(update, inp, chatbox)
        stop.click(None, None, None, cancels=[click_event])
        #inp.submit(user, inp, inp).then(update, inp, chatbox)
        clear.click(lambda: (None, None), None, [inp, chatbox], queue=False)
# theme=gr.themes.Default().set(button_primary_border_color_dark=, hover)
demo.queue().launch(server_name='0.0.0.0', share=True, inbrowser=False, server_port=7001)
# demo = gr.Interface(update,
#                     inputs="text",
#                     outputs="html")

# demo_test = gr.Interface(update,
#                     inputs="textbox",
#                     outputs="html",
#                     examples=["Who is the brother of the 2022 NBA FMVP?"])
# demo_test.queue().launch(server_name='0.0.0.0', share=True, inbrowser=False, server_port=7001)

# import time
# import gradio as gr

# def test(message):
#     result = ""
#     for i in range(5):
#         time.sleep(0.3)
#         result += message
#         re_html = f"<span style='color: red'>{result}</span>"
#         yield re_html

# gr.Interface(test, "textbox", "html").queue().launch()