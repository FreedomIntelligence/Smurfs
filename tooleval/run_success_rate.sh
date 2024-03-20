export CONVERTED_ANSWER_PATH=tooleval/model_predictions_converted/
export SAVE_PATH=tooleval/success_rate_results
export CANDIDATE_MODEL=mistral_dfsdt_v5
export API_POOL_FILE=tooleval/keys.json

python tooleval/eval_success_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids tooleval/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 1
