cuda_devices=$CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export RUN_SLOW=true
# export ACCELERATE_USE_DEEPSPEED=true

deepspeed --include localhost:$cuda_devices --master_port 24178 train_sdxl_stage_1_deepspeed.py --config configs/train/stage1_sdxl_21k_pose1024.yaml "$@"