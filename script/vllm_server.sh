model_path="/mntcephfs/lab_data/chenjunzhi/Mistral-7B-Instruct-v0.2"
model_name="Mistral-7B-Instruct-v0.2"
tensor_parallel_size=2

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size
