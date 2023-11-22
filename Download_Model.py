from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import shutil

# Define base path and model directory
base_path = os.getcwd()
model_dir = os.path.join(base_path, "models")

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

def download_model(model_name):
    """
    Download a model and its tokenizer from Hugging Face if they are not already downloaded.
    Args:
    model_name (str): The name of the model on Hugging Face.
    """
    # Format the model name to replace '--' with '_'
    formatted_model_name = model_name.replace('--', '_')
    formatted_model_name = model_name.replace('/', '_')
    target_dir = os.path.join(model_dir, f"{formatted_model_name}")

    # Check if the model already exists
    if os.path.exists(target_dir):
        print(f"Model '{model_name}' already downloaded.")
        return

    print(f"Downloading: {model_name}")

    # Download the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision="main",
        cache_dir=model_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=model_dir,
        model_max_length=4096,
        truncation_side='left'
    )

    # Save the model and tokenizer to a directory
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    # Rename the folder
    new_dir = os.path.join(model_dir, f"{formatted_model_name}")
    shutil.move(target_dir, new_dir)

    print(f"Model and tokenizer for '{model_name}' downloaded and saved in '{new_dir}'.")

if __name__ == "__main__":
    # Read model name from settings.json
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)
        model_name_or_path = settings['model_name_or_path']

    # Check if the model directory already exists
    model_path = os.path.join(model_dir, model_name_or_path.replace('--', '_'))
    if not os.path.exists(model_path):
        # Download the specified model if it doesn't exist
        download_model(model_name_or_path)
    else:
        print(f"Model {model_name_or_path} already exists.")

