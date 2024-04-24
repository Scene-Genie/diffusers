export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="scene-genie/train-reflections-shadows-filtered"
export OUTPUT_DIR="p2p-scenegenie-filtered"
export VAL_IMAGE_URL="https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/val-image-1.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/val-image-2.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/val-image-3.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/val-image-4.jpg"


accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --output_dir="model_out" \
  --original_image_column="original_image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="prompt" \
  --num_train_epochs=3 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=1 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url=$VAL_IMAGE_URL \
  --validation_prompt="add reflection to the can,add reflection to the can,add shadow to the can,add shadow to the can" \
  --validation_epochs=1 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub