export CONVERTED_ANSWER_PATH=tooleval/model_predictions_converted/
export SAVE_PATH=tooleval/preference_results
export PASS_TARE_PATH=tooleval/pass_rate_results
export REFERENCE_MODEL=chatgpt_cot
export CANDIDATE_MODEL=mistral_dfsdt_v5
export API_POOL_FILE=tooleval/keys.json

python tooleval/eval_preference.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids tooleval/test_query_ids \
    --save_path ${SAVE_PATH} \
    --pass_rate_result_path ${PASS_TARE_PATH} \
    --max_eval_threads 20 \
    --use_pass_rate true \
    --evaluate_times 4
