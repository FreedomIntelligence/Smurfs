export rapidapi_key=""
export toolbench_key=""
export OPENAI_KEY=""
export PYTHONPATH=./

model_name="Your_Model_Name"

python inference/inference.py \
    --model_name $model_name \
    --toolbench_key $toolbench_key \