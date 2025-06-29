# One-Click LLM Fine-Tuning in Google Colab

## Description

This repository provides a one-click solution to fine-tune a Language Model (LLM) using Google Colab. Leveraging [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685), it enables efficient fine-tuning on custom datasets. The default example uses the [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) model with a synthetic drone action dataset, but you can easily adapt it to your own use case.

## Features

- **One-Click Setup:** Run the notebook directly in Google Colab with minimal configuration.
- **Custom Dataset Support:** Use your own data in JSON format or adapt the provided example.
- **Efficient Fine-Tuning:** Utilizes LoRA for low-resource fine-tuning.
- **Resume Training:** Continue from saved checkpoints seamlessly.

## Installation

To get started, open the provided Google Colab notebook and run the following commands to install dependencies:

```bash
!pip install transformers peft torch ijson gdown
!pip install -U datasets
```

Ensure you have a Google Drive account to save checkpoints and outputs.

## Usage

1. **Open the Notebook:**
   - Access the Colab notebook via the provided link.

2. **Download or Upload Dataset:**
   - The default dataset (`synthetic_data.json`) is downloaded from Google Drive. Replace it with your own by updating the `drive_link` variable or uploading a file.

3. **Configure the Model:**
   - Default model: `SmolLM2-135M-Instruct`. Change `model_name` to your preferred LLM from Hugging Face.

4. **Run the Notebook:**
   - Execute all cells to download data, load the model, tokenize the dataset, and start training.

## Dataset

- **Default Dataset:** A synthetic JSON file (`synthetic_data.json`) with drone action examples (`vx, vy, vz, yaw`).
- **Custom Dataset:** Replace with your own JSON file. Expected format:
  ```json
  [{"key": "value", "response": "desired output"}, ...]
  ```
- Update the `format_example` function in the notebook to match your data structure.

## Model and Tokenizer

- **Default Model:** `SmolLM2-135M-Instruct` from Hugging Face.
- **Tokenizer:** Automatically loaded with the model. Padding token is set to the end-of-sequence token.
- **Customization:** Modify `model_name` in the notebook to use a different LLM.

## Training Configuration

- **LoRA Settings:**
  - `r`: 16 (rank of adaptation matrices)
  - `lora_alpha`: 32 (scaling factor)
  - `target_modules`: `["q_proj", "v_proj"]` (attention layers to adapt)
  - `lora_dropout`: 0.05
  - `bias`: "none"
  - `task_type`: "CAUSAL_LM"

- **Training Arguments:**
  - `output_dir`: "/content/drive/MyDrive/finetuned_model" (customize as needed)
  - `per_device_train_batch_size`: 4
  - `gradient_accumulation_steps`: 4
  - `max_steps`: 2000 (adjust based on dataset size)
  - `learning_rate`: 1e-4
  - `fp16`: True (for GPU efficiency)
  - `save_steps`: 200
  - `eval_steps`: 50
  - See the notebook for full details and tweak as required.

## Resuming Training

- The script checks for checkpoints in `output_dir`. If found, training resumes automatically.
- To resume from a specific checkpoint, set `resume_from_checkpoint` to the checkpoint path (e.g., `checkpoint-1000`).

## Saving the Model

- Fine-tuned model and tokenizer are saved to `/content/drive/MyDrive/lora_adapters` (customizable).
- Use these for inference or further fine-tuning.

## License

[Add your preferred license here, e.g., MIT License]