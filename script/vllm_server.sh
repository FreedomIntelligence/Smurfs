model_path="Your/Model/Path"
model_name="Your/Model/Name"
tensor_parallel_size=4

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size
