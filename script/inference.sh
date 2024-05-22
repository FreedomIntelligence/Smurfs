export rapidapi_key=""
export toolbench_key="lDPxqwOYRltCORiQt4viGwgAehsDt2pn5ez4W78hwK5nVtyPET"
export OPENAI_KEY=""

model_name="Mistral-7B-Instruct-v0.2"
method_name="smurfs"
test_query_id_path="/mntcephfs/lab_data/chenjunzhi/data/test_query_ids"
query_file_dir="/mntcephfs/lab_data/chenjunzhi/data/test_instruction"
tool_env_dir="/mntcephfs/lab_data/chenjunzhi/data/toolenv/tools"


python Smurfs/inference/inference.py \
    --model_name $model_name \
    --toolbench_key $toolbench_key \
    --method_name $method_name \
    --test_query_id_path $test_query_id_path \
    --query_file_dir $query_file_dir \
    --tool_env_dir $tool_env_dir \