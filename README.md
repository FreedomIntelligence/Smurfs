<div align="center">
<h1>Smurfs<br><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3602.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3607.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3623.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3625.gif></a></h1>
</div>

<p align="center">
<img src="assets/logo.webp" width="512">
</p>

🤖This project aims to construct a synergistic multi-agent system that can handle complex multi-tool instructions without necessitating extra training. This MAS system is called Smurfs, just like the beloved cartoon characters of the same name, symbolize unity and resourcefulness, and are good at
using tools to overcome any challenge they encounter.

## ✨ What's New
+  [2024.05.23] We release Smurfs, a multi-agent framework that gives LLM access to external tools to efficiently
solve complex tasks.
   + The code and datasets are available at [Smurfs](#).
   + The paper is available at [Smurfs: Leveraging Multiple Proficiency Agents with Context-Efficiency
for Tool Planning](http://arxiv.org/abs/2405.05955).

## 🗓 Coming Soon
- [x] Code release of our [paper](http://arxiv.org/abs/2405.05955)
- [ ] Support customised API inference
- [ ] Support CLI and GUI inference

✨Here is an overview of the Smurfs framework.

<br>
<div align="center">
<img src="assets/overview.png" width="800px">
</div>
<br>

## 📚 Data
You need to first get the toolbench dataset using the following link: [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/) to do the experiment. 

The reproduction data of smurfs can be found at . You can use these data to reproduce our experiment result.

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
example_dir="path to your example data"

python Smurfs/data/post_process.py \
    --input_dir $input_dir \
    --test_sets "${test_sets[@]}" \
    --example_dir $example_dir
```
- Evaluation:
For Evaluation process, download tooleval from [tooleval](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval) and use the post-processd data as the CONVERTED_ANSWER to do the evaluation following [tooleval](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval).

## 📊 Experiment Result

In our main experiments on toolbench, Smurfs can improve the ability of the base model to handle complex multi-tool instructions that match or even exceed that of capabilities of GPT4-DFSDT. Below are the main results. The win rate for each model is compared with ChatGPT-ReACT.

**Pass Rate:**
| **Method** | **I2 Category** | **I2 Instruction** | **I3 Instruction** | **Average** |
| --- | --- | --- | --- | --- |
| ToolLLama-7B (ReACT) | 31.5 | 30.5 | 25.0 | 29.0 |
| ToolLLama-7B (DFSDT) | 77.0 | 77.0 | 66.0 | 73.3 |
| Vicuna-7B (ReACT&DFSDT) | 0.0 | 0.0 | 0.0 | 0.0 |
| Vicuna-7B (Smurfs) | 70.5 | 77.0 | 78.0 | 75.2 |
| Mistral-Instruct-7B (ReACT&DFSDT) | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-Instruct-7B (Smurfs) | **79.5** | 77.5 | **79.0** | **78.7** |
| GPT4 (ReACT) | 72.0 | 67.0 | 47.0 | 62.0 |
| GPT4 (DFSDT) | 77.5 | **79.5** | 71.0 | 76.0 |
| GPT4 (Smurfs) | 72.0 | 71.0 | 64.0 | 69.0 |

**Win Rate:**
| **Method** | **I2 Category** | **I2 Instruction** | **I3 Instruction** | **Average** |
| --- | --- | --- | --- | --- |
| ToolLLama-7B (ReACT) | 41.8 | 50.8 | 55.0 | 49.2 |
| ToolLLama-7B (DFSDT) | 58.0 | 68.5 | 69.0 | 65.2 |
| Vicuna-7B (ReACT&DFSDT) | 0.0 | 0.0 | 0.0 | 0.0 |
| Vicuna-7B (Smurfs) | 64.25 | 73.0 | 87.0 | 74.8 |
| Mistral-Instruct-7B (ReACT&DFSDT) | 0.0 | 0.0 | 0.0 | 0.0 |
| Mistral-Instruct-7B (Smurfs) | **79.2** | **80.0** | **94.0** | **84.4** |
| GPT4 (ReACT) | 60.3 | 65.8 | 78.0 | 68.0 |
| GPT4 (DFSDT) | 63.3 | 73.3 | 84.0 | 73.5 |
| GPT4 (Smurfs) | 77.0 | 77.5 | 89.5 | 81.3 |

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
- [Least to most prompt](https://github.com/RUCAIBox/LLMBox.git)
