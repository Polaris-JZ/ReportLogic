pip install -e .
pip install math-verify
pip install python-Levenshtein
pip install -U transformers
# deepspeed 0.16 has a memory leak; downgrade to 0.15.0 before installing vllm 0.7.2 deps
pip install deepspeed==0.15.0
pip install -U nvidia-cublas-cu12
pip install timeout_decorator
pip install vllm==0.8.5
pip install torch==2.6.0
pip install torchaudio==2.6.0 torchvision==0.21.0
pip install nltk
pip install langdetect
