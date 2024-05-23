<div align="center">
<h1>Smurfs<br><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3602.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3607.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3623.gif></a><a href=https://yoursmiles.org/h-smurf.php><img src=https://yoursmiles.org/hsmile/smurf/h3625.gif></a></h1>
</div>

<p align="center">
<img src="assets/logo.webp" width="512">
</p>

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

✨✨Features:

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


## Citation

## Acknowledgement

## Contact


