# Convert a Grouped-GEMM Qwen3 MoE checkpoint to HF format.
# Fill in --input_dir / --output_dir / --memo to match your run.

python modeling4qwen3moe/convert_gemm_to_hf.py \
    --input_dir /path/to/checkpoint/grouped_gemm \
    --output_dir /path/to/checkpoint/hf \
    --memo my_run
