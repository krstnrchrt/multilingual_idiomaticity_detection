#!/bin/sh

# Set Weights & Biases configuration
echo "Setting up W&B environment variables..."
export WAND_API_KEY="$(cat /slid/secrets/.wandb_api_key)"
export WAND_USERNAME="$(cat /slid/secrets/.wandb_username)"
export WANDB_ENTITY="anlp-slid"
export WANDB_ANONYMOUS="never"
export WANDB_PROJECT="slid"

# Run the real experiment
echo "Starting main.py to conduct experiments..."
python -m src.main
echo "Finished with main.py. "
