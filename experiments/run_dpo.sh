GRAD_ACCUMULATION_STEPS=1
DATASET_NAME=h
BATCH_SIZE=32
EVAL_BATCH_SIZE=4
SFT_MODEL_PATH=""
OUTPUT_DIR=""
SEEDS=('1' '2' '3')
BETAS=('0.05')

for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do

OUTPUT_DIR=src/models/finetuned_models/dpo/${SFT_MODEL_PATH}-dpo_beta_${BETA}_${DATASET_NAME}_${SEED}

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port=6500\
    cli/train_reward.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --eval_every=51200 \
    --exp_name=${SFT_MODEL_PATH}_reward_${DATASET_NAME}\
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=150 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR

done
done