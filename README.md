<div align="center">
<h1>Smurfs<br><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3602.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3607.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3623.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3625.gif></a></h1>
</div>

<p align="center">
<img src="assets/logo.webp" width="512">
</p>

🤖This project aims to construct a synergistic multi-agent system that can handle complex multi-tool instructions without necessitating extra training. This MAS system is called Smurfs, just like the beloved cartoon characters of the same name, symbolize unity and resourcefulness, and are good at
using tools to overcome any challenge they encounter.

## ✨ What's New
+  [2025.04.29] Smurfs is accepted by NAACL 2025! Final version of paper can be seen [here](https://aclanthology.org/2025.naacl-long.169.pdf)
+  [2024.09.08] You can try Smurfs on huggingface space [here](https://huggingface.co/spaces/szjiozi/Smurfs)
+  [2024.07.05] CLI and GUI inference have been supported.
+  [2024.06.25] HotpotQA evaluation has been supported.
+  [2024.06.25] We release the new version of our paper at [here](http://arxiv.org/abs/2405.05955)
+  [2024.05.23] We release Smurfs, a multi-agent framework that gives LLM access to external tools to solve complex tasks efficiently.
   + The code and data are available at [Smurfs](#).

## 🗓 Coming Soon
- [x] Code release of our [paper](http://arxiv.org/abs/2405.05955)
- [x] Support customized API inference
- [x] Support CLI inference
- [x] Support GUI inference
- [ ] More tools are coming

✨Here is an overview of the Smurfs framework.

<br>
<div align="center">
<img src="assets/overview.png" width="800px">
</div>
<br>

✨✨Here is a demo of using Smurfs

<div align="center">



https://github.com/FreedomIntelligence/Smurfs/assets/99324175/2edd6d2e-e7f1-4e8e-a78e-56c613d2ba13



</div>

✨✨You can also try it using our huggingface space [here](https://huggingface.co/spaces/szjiozi/Smurfs)

## 🚀 Inference
- CLI Inference:

Add tool function to Smurfs/tools/tool_env.py and add all available tool function to tool_env variable, for example:
```python
class HotpotToolEnv: ...

HPEnv = HotpotToolEnv()

tool_env = {
    "BingSearch": HPEnv.BingSearch,
    "Retrieve": HPEnv.Retrieve,
    "Lookup": HPEnv.Lookup
    }
```
Then add the tool description to a json file, for example:
```json
[
    {
        "api_name": "BingSearch",
        "api_description": "BingSearch can search for rich external knowledge on the Internet based on keywords, which can compensate for knowledge fallacy and knowledge outdated.",
        "required_parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "query used to search on the Internet. Should be specific and precise with your query to increase the chances of getting relevant results.",
                "default": ""
            }
        ],
        "optional_parameters": []
    },
   ... 
]
```
then run
```bash
python Smurfs/deploy/cli_inference.py
```
and type in the input query.

- GUI Inference:
Follow the same steps as in CLI inference to prepare the tools, then run
```python
python Smurfs/deploy/gradio_inference.py
```
  
## 📚 Data
You need to first get the StableToolBench dataset and server cache by following the instructions in their [repo](https://github.com/THUNLP-MT/StableToolBench.git), and deploy the API server to perform the experiment.

The reproduction data of smurfs can be found at [reproduction_data](https://github.com/FreedomIntelligence/Smurfs/tree/main/reproduction_data). You can use these data to reproduce our experiment result.

## 🧐 Experiment
- Launch vLLM server:
Using the script in Smurfs/script/vllm_server.sh to launch a vLLM server of the model that you want to use in the experiment. Suppose you use Mistral-7B-Instruct-v0.2 to do the experiment, you use 4 GPUs to launch the vLLM server and the model is saved at /home/Mistral-7B-Instruct-v0.2, the script looks like:
```bash
model_path="/home/Mistral-7B-Instruct-v0.2"
model_name="Mistral-7B-Instruct-v0.2"
tensor_parallel_size=4

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size
```

Noted that some models do not have chat template in their tokenizer config file like vicuna, you need to download their chat template from the internet (for example [here](https://github.com/chujiezheng/chat_templates.git)) and use the script below:
```bash
model_name="Your/Model/Name"
tensor_parallel_size=4
chat_template_path="Your/Template/Path"

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size --chat-template $chat_template_path
```

The vLLM server can provide easy, fast, and cheap LLM serving for most popular open-source models. Using it can significantly increase the experiment speed. For more information of vLLM, see [vLLM](https://github.com/vllm-project/vllm.git)
  
- Inference:
To use the toolbench apis with the toolbench server, you need to first get your toolbench_key (More information can be seen [here](https://github.com/OpenBMB/ToolBench.git)) and pass it through `toolbench_key`. Suppose you save the toolbench data in the directory toolbench_data/data/, the script looks like:
```bash
export toolbench_key="Your_key"

model_name="Mistral-7B-Instruct-v0.2"
method_name="smurfs"
test_query_id_path="toolbench_data/data/test_query_ids"
query_file_dir="toolbench_data/data/test_instruction"
tool_env_dir="toolbench_data/data/toolenv/tools"


python Smurfs/inference/inference.py \
    --model_name $model_name \
    --toolbench_key $toolbench_key \
    --method_name $method_name \
    --test_query_id_path $test_query_id_path \
    --query_file_dir $query_file_dir \
    --tool_env_dir $tool_env_dir
```
If you want to do inference with customized RapidAPI account, pass your rapidapi key through rapidapi_key and specify the `use_rapidapi_key` argument in the script:
```bash
export rapidapi_key="Your_key"

model_name="Mistral-7B-Instruct-v0.2"
method_name="smurfs"
test_query_id_path="toolbench_data/data/test_query_ids"
query_file_dir="toolbench_data/data/test_instruction"
tool_env_dir="toolbench_data/data/toolenv/tools"


python Smurfs/inference/inference.py \
    --model_name $model_name \
    --toolbench_key $toolbench_key \
    --method_name $method_name \
    --test_query_id_path $test_query_id_path \
    --query_file_dir $query_file_dir \
    --tool_env_dir $tool_env_dir \
    --use_rapidapi_key
```
- Post Process:
The output of your experiment will be saved at Smurfs/data/your_method_name/. You need to post process it using the following script so that the tooleval from toolbench can evaluate its pass rate and win rate:
```bash
test_sets=("G2_category" "G2_instruction" "G3_instruction")
input_dir="data/smurfs"
example_dir="reproduction_data/mistral_smurfs"

python Smurfs/data/post_process.py \
    --input_dir $input_dir \
    --test_sets "${test_sets[@]}" \
    --example_dir $example_dir
```
- Evaluation:
For Evaluation process, download tooleval from [tooleval](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval) and use the post-processd data as the CONVERTED_ANSWER to do the evaluation following [tooleval](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval).

## 📊 Experiment Result

In our main experiments on StableToolBench, Smurfs can improve the ability of the base model to handle complex multi-tool instructions that match or even exceed that of capabilities of GPT4-DFSDT. Below are the main results. The win rate for each model is compared with ChatGPT-ReACT.

**Pass Rate:**
| Backbone | Method | I1-Inst. | I1-Cat. | I1-Tool. | I2-Cat. | I2-Inst. | I3-Inst. | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-3.5 Turbo | ReACT | 41.6±1.2 | 48.4±0.5 | 52.5±0.5 | 52.2±1.0 | 31.6±1.2 | 39.9±2.0 | 44.4±1.1 |
| GPT-3.5 Turbo | DFSDT | 54.1±1.0 | 60.1±0.0 | 59.9±1.7 | 60.9±0.9 | 52.8±3.7 | 44.3±4.8 | 55.4±2.0 |
| GPT-3.5 Turbo | **Smurfs** | 60.3±1.5 | 67.0±1.0 | 60.3±1.3 | 54.3±0.4 | 42.6±1.6 | 60.1±1.0 | 57.4±1.1 |
| Mistral-7B | ReACT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-7B | DFSDT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-7B | **Smurfs** | **76.3±0.8** | **86.7±1.2** | **81.0±1.9** | **70.4±2.7** | **63.8±2.4** | **85.2±0.7** | **77.2±1.6** |
| GPT-4 Turbo | ReACT | 41.1±1.5 | 53.2±1.3 | 42.2±1.1 | 50.0±0.7 | 38.7±0.8 | 37.7±1.3 | 43.8±1.1 |
| GPT-4 Turbo | DFSDT | 52.7±1.4 | 58.2±0.9 | 59.7±1.2 | 59.3±0.7 | 52.2±2.3 | 61.5±1.8 | 57.3±1.4 |
| GPT-4 Turbo | **Smurfs** | 59.3±1.4 | 73.3±1.3 | 67.4±0.7 | 66.7±1.9 | 55.5±1.4 | 70.5±0.0 | 65.5±1.1 |

**Win Rate:**
| Backbone | Method | I1-Inst. | I1-Cat. | I1-Tool. | I2-Cat. | I2-Inst. | I3-Inst. | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-3.5 Turbo | ReACT | / | / | / | / | / | / | / |
| GPT-3.5 Turbo | DFSDT | 64.4 | 61.4 | 53.8 | 62.9 | 66.0 | 54.1 | 60.4 |
| GPT-3.5 Turbo | **Smurfs** | 65.0 | 69.9 | 54.4 | 63.7 | 64.2 | 57.4 | 62.4 |
| Mistral-7B | ReACT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-7B | DFSDT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-7B | **Smurfs** | 63.8 | 62.7 | 58.2 | 54.0 | 67.0 | 57.4 | 60.5 |
| GPT-4 Turbo | ReACT | 60.1 | 62.1 | 48.1 | 57.3 | 65.1 | 47.5 | 56.7 |
| GPT-4 Turbo | DFSDT | 69.9 | 66.0 | 58.2 | 62.1 | **67.9** | 65.6 | 65.0 |
| GPT-4 Turbo | **Smurfs** | **71.2** | **72.5** | **69.6** | **73.4** | 66.0 | **72.1** | **70.8** |

## Citation
```
@misc{chen2024smurfs,
      title={Smurfs: Leveraging Multiple Proficiency Agents with Context-Efficiency for Tool Planning}, 
      author={Junzhi Chen and Juhao Liang and Benyou Wang},
      year={2024},
      eprint={2405.05955},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ) and the Shenzhen Rsearch Institute of Big Data (SRIBD).

## Acknowledgement
We are aware that our works are inspired by the following works, including but not limited to
- [Toolbench](https://github.com/OpenBMB/ToolBench.git)
- [StableToolBench](https://github.com/THUNLP-MT/StableToolBench.git)
- [Least to most prompt](https://github.com/RUCAIBox/LLMBox.git)
