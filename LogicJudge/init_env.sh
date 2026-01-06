pip install grouped-gemm==0.1.4
pip install vllm==0.8.5
pip install nltk
pip install langdetect
pip install torchdata
pip install ray[default]
pip install -U deepspeed
pip install numpy==1.26.3
pip install megablocks==0.9.0

/bin/cp -f vllm4ling/models/bailing_moe.py /opt/conda/lib/python3.10/site-packages/vllm/model_executor/models
/bin/cp -f vllm4ling/models/registry.py /opt/conda/lib/python3.10/site-packages/vllm/model_executor/models
/bin/cp -f vllm4ling/configs/__init__.py /opt/conda/lib/python3.10/site-packages/vllm/transformers_utils/configs
/bin/cp -f vllm4ling/configs/bailing_moe.py /opt/conda/lib/python3.10/site-packages/vllm/transformers_utils/configs
/bin/cp -f dockerfile/torch_checkpoint_engine.py /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py # patch deepspeed 0.15 for torch2.6.0
/bin/cp -f dockerfile/grouped_gemm_moe.py  /opt/conda/lib/python3.10/site-packages/atorch/modules/moe/grouped_gemm_moe.py
