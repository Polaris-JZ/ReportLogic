export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

python generate_adv_cklist.py \
  --adversarial-jsonl /path/to/project/bias/data_bias/adversarial.jsonl \
  --out-jsonl /path/to/project/bias/data_bias/adversarial_cklist.csv \
  --resume
