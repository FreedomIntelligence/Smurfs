model_path="/path/to/your/model"
model_name="Your_model_name"
tensor_parallel_size=Your_GPU_number

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size
