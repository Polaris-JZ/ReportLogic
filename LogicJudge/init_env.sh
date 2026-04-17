pip install grouped-gemm==0.1.4
pip install vllm==0.8.5
pip install nltk
pip install langdetect
pip install torchdata
pip install ray[default]
pip install -U deepspeed
pip install numpy==1.26.3
pip install megablocks==0.9.0

# patch deepspeed 0.15 for torch 2.6.0
/bin/cp -f dockerfile/torch_checkpoint_engine.py /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py
