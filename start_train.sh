base_dir=/home/weipeilun/playground/racing/LLaMA-Factory
cd $base_dir
export PYTHONPATH=$base_dir:$base_dir/src:$base_dir/data:$PYTHONPATH
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python $base_dir/src/llamafactory/cli.py train examples/train_lora/qwen2_vl.yaml