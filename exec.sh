export MODEL_NAME="SG161222/Realistic_Vision_V2.0"
export INSTANCE_DIR="input"
export CLASS_DIR="class_input"
export OUTPUT_DIR="output"
export LOGGING_DIR="logs"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_name_or_path "stabilityai/sd-vae-ft-mse" \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --center_crop \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --class_promp="a photo of a person in a restaurant" \
  --center_crop \
  --num_class_images=100 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --train_text_encoder \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2500 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --resolution=512 \
  --logging_dir=$LOGGING_DIR
