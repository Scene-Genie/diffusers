export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="scene-genie/instagram-dataset-train"
export OUTPUT_DIR="p2p-scenegenie"
export VAL_IMAGE_URL="https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/image1.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/image2.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/image3.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/image4.jpg,https://raw.githubusercontent.com/Scene-Genie/finetune_instructp2p/main/image5.jpg"


accelerate launch --mixed_precision="fp16" train_instruct_pix2pix_with_prompt.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --output_dir="model_out" \
  --original_image_column="resized_untouched_image" \
  --edited_image_column="resized_touched_image" \
  --edit_prompt_column="prompt" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 \
  --checkpoints_total_limit=1 \
  --learning_rate=5e-05 \
  --conditioning_dropout_prob=0.05 \
  --max_grad_norm=1 \
  --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url=$VAL_IMAGE_URL \
  --validation_prompt="" \
  --validation_epochs=1 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub