nvidia-smi
SFT_MODEL_PATH=""
GRAD_ACCUMULATION_STEPS=8
DATASET_NAME=hh
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
SEEDS=('1' '2' '3')
BETAS=('0.05')

for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do

OUTPUT_DIR=src/models/finetuned_models/is_dpo/Llama-3.2-${SFT_MODEL_PATH}-is_dpo_beta_${BETA}_${DATASET_NAME}_${SEED}

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port=5500\
    cli/train_is_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eps=1.0 \
    --eval_every=19200 \
    --beta=${BETA} \
    --exp_name=${SFT_MODEL_PATH}_is_dpo_beta_${BETA}_${DATASET_NAME}_pref_seed_${SEED} \
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