cd /home/weipeilun/playground/racing/LLaMA-Factory
PYTHONPATH=/home/weipeilun/playground/racing/LLaMA-Factory:$PYTHONPATH
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/data/anaconda3/envs/llama-factory/bin/python /home/weipeilun/playground/racing/LLaMA-Factory/src/llamafactory/cli.py train examples/train_lora/qwen2_vl.yaml > train.log 2>&1
