# export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="scene-genie/train-shadow-blocked"
export OUTPUT_DIR="scenegenie-shadow-add-sdxl"
export VAL_IMAGE_URL="https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/val-image-1.jpg"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"


accelerate launch train_instruct_pix2pix_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_ID \
  --resolution=512 \
  --seed=42 \
  --original_image_column="original_image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="prompt" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="add shadow to the can" \
  --validation_steps=5 \
  --checkpointing_steps=5000 \
  --output_dir=$OUTPUT_DIR \
  --push_to_hub