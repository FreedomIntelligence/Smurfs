model_name="Your/Model/Name"
tensor_parallel_size=4
chat_template_path="Your/Template/Name"

cd $model_path
cd ..
python -m vllm.entrypoints.openai.api_server --model $model_name --dtype=half --tensor-parallel-size $tensor_parallel_size --chat-template $chat_template_path