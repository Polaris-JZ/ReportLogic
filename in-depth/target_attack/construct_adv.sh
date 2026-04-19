python3 construct_adversarial.py \
  --input-csv /path/to/project/data/deep_research/GT/deep_research_gt.csv /path/to/project/data/quora/GT/quora_gt.csv /path/to/project/data/zhihu/GT/zhihu_gt.csv\
  --output-path /path/to/project/bias/data/adversarial.jsonl \
  --sample-count 100 \
  --attack-strengths low,medium,high \
  --language-retry-limit 10 \
  --resume \
  --dedupe-output \
  --prune-mismatched-output