cd /root/my_env/LLaMA-Factory
PYTHONPATH=/root/my_env/LLaMA-Factory:$PYTHONPATH
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/data/anaconda3/envs/llama-factory/bin/python /root/my_env/LLaMA-Factory/src/llamafactory/cli.py train examples/train_lora/qwen2_vl_luchentec.yaml > train.log 2>&1
