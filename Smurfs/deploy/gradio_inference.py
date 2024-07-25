import gradio as gr

import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Smurfs.inference.smurfs_worker import smurfs_hotpot_worker, smurfs_worker, stream_smurfs_worker
# from Smurfs.tools.tool_env import HotpotToolEnv
from Smurfs.deploy import global_dict
from Smurfs.tools.tool_env import tool_env
from Smurfs.model.openai_model.openai_model import OpenAI_Model, OpenRouter_Model
from Smurfs.agents.answer_agent.answer import stream_answer_agent
from Smurfs.agents.executor_agent.executor import stream_executor_agent
from Smurfs.agents.planning_agent.planner import stream_hotpot_planning_agent
from Smurfs.agents.verifier_agent.verifier import stream_verifier_agent

from Smurfs.tools.docqa.api import tool_env as docqa_tool_env
from Smurfs.tools.hotpotQA.api import tool_env as hotpot_tool_env
from Smurfs.tools.math.api import tool_env as math_tool_env
from Smurfs.tools.shell.api import tool_env as shell_tool_env
from Smurfs.tools.websearch.api import tool_env as websearch_tool_env


import json
import threading
import joblib
from tqdm import tqdm
import time

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from datetime import datetime
current_datetime = datetime.now()
# user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
# inp = gr.Textbox(placeholder="Enter your task")
# css = """
# .btn {background-color: blue; color: white;}
# #bot {background-color: blue; color: white;}
# #e {display: inline-block; vertical-align: middle;}
# """

# def update(name):
#     return f"<span style='color: red'>Welcome to Gradio, {name}!</span>"

tool_env_map = {
    "shell": shell_tool_env, 
    "math": math_tool_env, 
    "docqa": docqa_tool_env, 
    "hotpotQA": hotpot_tool_env, 
    "websearch": websearch_tool_env
}

total_env, env_name_list = {}, []

def loading():
    return "Loading..."

def load_text_from_pdf(up, key=None):
    global global_dict
    if key == None or key == '':
        key = os.environ.get("OPENAI_API_KEY")
    pdf_path = up.name
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    # embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    global_dict["knowledge_base"] = FAISS.from_texts(chunks, embeddings)
    return "upload success!"
    #return knowledge_base

def update(query, OPENAI_API_KEY, BING_SUBSCRIPT_KEY, WOLFRAMALPH_APP_ID, WEATHER_API_KEYS):
    global total_env, env_name_list
    # print(total_env)
    # print(BING_SUBSCRIPT_KEY)
    # print(WOLFRAMALPH_APP_ID)
    # print(WEATHER_API_KEYS)
    # model_name = "mistralai/mistral-7b-instruct-v0.2"
    model_name = "gpt-4"
    method_name = "cli_inference"
    tool_doc_path = "Smurfs/tools/tool_doc.json"
    if OPENAI_API_KEY == None or OPENAI_API_KEY == '':
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = OpenAI_Model(model_name=model_name, api_key=OPENAI_API_KEY)
    #llm = OpenRouter_Model(model_name=model_name)
    if "docqa" in total_env:
        sys_prompt = llm.sys_prompt + "You already have access to the file uploaded by the user. So just answer the question from the user, you don't need to find the file first."
        llm.change_sys_prompt(sys_prompt)
    else:
        llm.set_default_sys_prompt()
    # parser_llm = OpenAI_Model(model_name="gpt-4")
    with open(tool_doc_path, "r") as f:
        tool_doc = json.load(f)
    tool_doc["bing_search"]["api_description"] += f"Today is {current_datetime.year}.{current_datetime.month}.{current_datetime.day}"
    available_tools = []
    for env_name in env_name_list:
        available_tools.append(tool_doc[env_name])

    test_set = "cli"
        
    output_dir = f"data/{method_name}/{test_set}/answer"
    results_dir = f"data/{method_name}/{test_set}/results.json"
    if not os.path.exists(f"data/{method_name}/{test_set}/parser_log"):
        os.makedirs(f"data/{method_name}/{test_set}/parser_log")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # HP_answer_agent = answer_agent(llm=parser_llm, logger_dir=f"data/{method_name}/{test_set}/parser_log")
    # worker = smurfs_hotpot_worker(available_tools, HotpotToolEnv, llm, method_name, test_set, answer_agent, executor_agent,hotpot_planning_agent, verifier_agent)
    #print(total_env)
    print(BING_SUBSCRIPT_KEY)
    worker = stream_smurfs_worker(available_tools, total_env, llm, method_name, test_set, stream_answer_agent, stream_executor_agent, stream_hotpot_planning_agent, stream_verifier_agent, OPENAI_API_KEY, BING_SUBSCRIPT_KEY, WOLFRAMALPH_APP_ID, WEATHER_API_KEYS)
    stream_generator = worker.run(query, 0)
    # messages = []
    while True:
        try:
            response = next(stream_generator)
            messages = [(query, response)]
            yield messages
        except StopIteration:
            break

def update_tools(rs):
    global total_env, env_name_list
    total_env = {}
    env_name_list = []
    for tool_system in rs:
        tool = tool_system.split(": ")[0]
        env = tool_env_map[tool]
        print(f"env: {env}")
        for e in env:
            if e not in env_name_list:
                total_env[e] = env[e]
                env_name_list.append(e)
    print(total_env)
    #return total_env, env_name_list

def user(user_msg):
    return user_msg

tools = ["math: Tool that can handle mathematical problems", 
         "docqa: Tool that can answer questions about your uploaded file", 
         "hotpotQA: Tool that can do multi-hop commonsense reasoning", 
         "websearch: Tool that can do web search to answer your question"]
websearch_example = ["请根据深圳明天的天气推荐给我推荐一套穿搭方案，结果用中文输出。", "今年的中秋节是哪天？用中文输出"]
math_example = ["Calc integral of sin(x)+2x^2+3x+1 from 0 to 1", "When both sides of a right triangle are 6 and 8, what is the length of the other side?"]
inp = gr.Textbox(placeholder="Please input your task", label="Task")
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Smurfs</h1>""")
    #gr.Markdown("""<figure><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3602.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3607.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3623.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3625.gif></a></figure>""")    
    #gr.HTML("""<a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3602.gif>""")
    with gr.Row():
        with gr.Column(scale=1):
            inp.render()
            rs = gr.Dropdown(choices=tools, label="Tool Systems", multiselect=True)
            file_output = gr.File(file_types=[".pdf"])
            with gr.Accordion("Model and Keys", open=False):
                model_name = gr.Dropdown(label="Moel Name", choices=["gpt-3.5", "gpt-4o", "gpt-4"])
                openai_key = gr.Textbox(label="OpenAI API Key", placeholder="Please Enter Your OpenAI API Key")
                bing_search_key = gr.Textbox(label="BingSearch Key", placeholder="Please Enter Your BingSearch Key")
                wolframalpha_key = gr.Textbox(label="Wolframalpha API Key", placeholder="Please Enter Your WolframAplpha Key")
                weather_key = gr.Textbox(label="Weather API Key", placeholder="Please Enter Your Weather API Key")

                
            gr.Examples(["Who is the brother of the 2022 NBA FMVP?", "How much older is Lebron James than his oldest son?", "Calc integral of sin(x)+2x^2+3x+1 from 0 to 1", "Calculate the length of the hypotenuse of a right triangle when the other two sides are 6 and 8", "请根据深圳明天的天气推荐给我推荐一套穿搭方案，结果用中文输出。", "今年的中秋节是哪天？用中文输出"], inp)
            _submit = gr.Button("Submit")
            stop = gr.Button("Stop")
            clear = gr.Button("Clear")
            #upload = gr.UploadButton("Click to upload your pdf file")
            # btn = gr.Button("Run", elem_id="bot", elem_classes="btn")
        #with gr.Column(scale=1, elem_id="e"):
            # chatbox = gr.HTML()
        chatbox = gr.Chatbot(height=300)
        # btn.click(fn=update, inputs=inp, outputs=chatbox)
        file_output.upload(load_text_from_pdf, [file_output, openai_key], None)
        #upload.upload(loading, None, inp).then(load_text_from_pdf, upload, inp)
        rs.change(update_tools, rs, None)
        click_event = _submit.click(user, inp, inp).then(update, [inp, openai_key, bing_search_key, wolframalpha_key, weather_key], chatbox)
        stop.click(None, None, None, cancels=[click_event])
        #inp.submit(user, inp, inp).then(update, inp, chatbox)
        clear.click(lambda: (None, None), None, [inp, chatbox], queue=False)
    demo.load(
        None,
        None,
        _js="""
        () => {
        const params = new URLSearchParams(window.location.search);
        if (!params.has('__theme')){
            params.set('__theme', 'dark');
            window.location.search = params.toString();
        }
        }    
        """,
    )
demo.queue().launch(server_name='0.0.0.0', share=True, inbrowser=False, server_port=7001)
