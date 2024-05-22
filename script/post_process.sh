test_sets=("G2_category")
input_dir="data/smurfs"
example_dir="/home/chenjunzhi/JARVIS/easytool/data_toolbench/result/gpt-4_easytool"

python Smurfs/data/post_process.py \
    --input_dir $input_dir \
    --test_sets "${test_sets[@]}" \
    --example_dir $example_dir