export RAW_ANSWER_PATH=/home/chenjunzhi/ToolBench/toolbench/tooleval/model_predictions
export CONVERTED_ANSWER_PATH=/home/chenjunzhi/ToolBench/toolbench/tooleval/model_predictions_converted
export MODEL_NAME=toolllama_dfs
export METHOD=DFS_woFilter_w2
mkdir ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}

for test_set in G1_tool
do
    answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
    output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json
    
    python convert_to_answer_format.py\
        --answer_dir ${answer_dir} \
        --method ${METHOD} \
        --output ${output_file}
done

# for test_set in G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction
# do
#     answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
#     output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json
    
#     python convert_to_answer_format.py\
#         --answer_dir ${answer_dir} \
#         --method ${METHOD} \
#         --output ${output_file}
# done
