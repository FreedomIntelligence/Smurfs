export CONVERTED_ANSWER_PATH=tooleval/model_predictions_converted/
export SAVE_PATH=tooleval/pass_rate_results
export CANDIDATE_MODEL=mistral_dfsdt_v5
export API_POOL_FILE=tooleval/keys.json
#mistral_norepeat_least_to_most_dfsdt
python tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids tooleval/test_query_ids \
    --max_eval_threads 20 \
    --evaluate_times 4
