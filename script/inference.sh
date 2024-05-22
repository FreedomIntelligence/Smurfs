export rapidapi_key=""
export toolbench_key=""
export OPENAI_KEY=""

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
