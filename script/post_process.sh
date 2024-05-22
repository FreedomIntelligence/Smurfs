test_sets=("G2_category" "G2_instruction" "G3_instruction")
input_dir="data/smurfs"
example_dir="path to your example data"

python Smurfs/data/post_process.py \
    --input_dir $input_dir \
    --test_sets "${test_sets[@]}" \
    --example_dir $example_dir
