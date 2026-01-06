from aistudio_checkpoint.checkpoint_supporter import sync_checkpoint

sync = "MYUN" # WULANCHABU, MALAYSIA, HY, HUANAN, MYUN
# sync_checkpoint("aistudio://6890462//input/chilin.fcl/exp/sft/r1/sft_0108_filter_ggn_long_cot_mixed/latest-unsharded_llama3/worker_0", sync)
sync_checkpoint("aistudio://7210596//input/chilin.fcl/kuiliu.lk/reasoning/rlhf/qwen25-7b-instruct_cold_start/worker_0", sync)
# sync_checkpoint("aistudio://7151159//path/to/llm_pretrain/model_hub/Qwen2.5-7B-Instruct", sync)