import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
model_dir = snapshot_download('qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4', cache_dir='./model', revision='master')
