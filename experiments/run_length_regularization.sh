nvidia-smi

SFT_MODEL_PATH=""
GRAD_ACCUMULATION_STEPS=4
DATASET_NAME=hh
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
SEEDS=('1' '2' '3')
BETAS=('0.01' '0.05' '0.1')

for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do

OUTPUT_DIR=models/length_dpo/${SFT_MODEL_PATH}-length_dpo_beta_${BETA}_${DATASET_NAME}_${SEED}

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port=5500\
    cli/train_length_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=12480 \
    --beta=${BETA} \
    --exp_name=${SFT_MODEL_PATH}_length_dpo_beta_${BETA}_${DATASET_NAME}_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=150 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 

done
done