from huggingface_hub import create_repo, upload_folder
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")

    parser.add_argument(
            "--output_dir",
            type=str,
            default="instruct-pix2pix-model",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")

    args = parser.parse_args()

    return args

args = parse_args()

unet = unwrap_model(unet)

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    text_encoder=unwrap_model(text_encoder),
    vae=unwrap_model(vae),
    unet=unet,
    revision=args.revision,
    variant=args.variant,
)
pipeline.save_pretrained(args.output_dir)

repo_id = create_repo(
                repo_id=Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


upload_folder(
    repo_id=repo_id,
    folder_path=args.output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"],
)