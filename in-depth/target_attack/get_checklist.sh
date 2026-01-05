export ANT_OPENAI_API_KEY_V2=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

python generate_adv_cklist.py \
  --adversarial-jsonl /path/to/project/bias/data/adversarial.jsonl \
  --out-jsonl /path/to/project/bias/data/adversarial_cklist.csv \
  --resume
