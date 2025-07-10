 python scripts/model_merger.py --backend fsdp \
     --local_dir /home/original_models/chengzong/sky-rl/stage1/oh-training-qwen3-8b-sft-filtered/oh-training-qwen3-8b/global_step_60/actor/ \
     --target_dir /home/original_models/chengzong/sky-rl-exp-qwen3-8b-sft-rl-step-60 \
     --hf_model_path /home/original_models/chengzong/qwen3-8b-sft-lamma-factory-1200