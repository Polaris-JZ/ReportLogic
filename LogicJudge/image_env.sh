pip install -e .
pip install math-verify
pip install python-Levenshtein
pip install -U transformers
# deepspeed在0.16引入了一个memory leak的bug, 先降级到0.15.0，然后再安装vllm 0.7.2的依赖
pip install deepspeed==0.15.0
pip install -U nvidia-cublas-cu12
pip install -U aistudio-checkpoint --index=https://pypi.antfin-inc.com/simple
pip install timeout_decorator
pip install aii-pypai -U --index=https://pypi.antfin-inc.com/simple
# deepspeed在0.16引入了一个memory leak的bug
# for ling-lite
pip install vllm==0.8.5
pip install torch==2.6.0
pip install torchaudio==2.6.0 torchvision==0.21.0
pip install atorch -U
pip install nltk
pip install langdetect