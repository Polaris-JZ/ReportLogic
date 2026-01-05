export ANT_OPENAI_API_KEY_V2=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# export ANT_OPENAI_API_KEY_V2=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python compute_atack_metrics.py \
  --input /path/to/project/bias/data_bias/adversarial_cklist.csv \
  --judge-model gpt-5-2025-08-07 gpt-5.1 o3-2025-04-16 claude-sonnet-4-5-20250929 claude-sonnet-4-20250514 qwen3-235b-a22b qwen3-max gemini-2.5-pro gemini-3-pro-preview claude-sonnet-4-5-20250929-think claude-sonnet-4-20250514-think gemini-2.5-pro-think gemini-3-pro-preview-think gpt-4.1-2025-04-14 qwen-max-2025-01-25 deepseek-v3-1-250821 \
  --num-workers 32 \
  --predictions-dir /path/to/project/bias/results_bias/ \
  --metrics-json /path/to/project/bias/results_bias/attack_metrics.json \
  --resume