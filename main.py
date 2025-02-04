# main.py
import argparse
import torch
from transformers import GenerationConfig, BitsAndBytesConfig

from data import load_and_prepare_datasets
from model import build_model_and_tokenizer, apply_lora
from train import get_training_args, train_model
from evaluate import evaluate_model

def main(args):
    # 1. Data Loading & Preparation
    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets()
    print("Datasets loaded and preprocessed.")

    # 2. Build the Model and Tokenizer
    base_model_name = "tiiuae/falcon-7b-instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model, tokenizer = build_model_and_tokenizer(base_model_name, bnb_config)
    print("Base model and tokenizer loaded.")

    # 3. Apply LoRA modifications to the model
    model = apply_lora(model)
    print("LoRA applied to the model.")

    # 4. Setup Training Arguments and Train the Model
    training_args = get_training_args(output_dir=args.output_dir)
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, training_args)
    print("Training complete.")

    # 5. Save Generation Configuration for Inference
    generation_config = GenerationConfig(
        max_length=512,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        length_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    generation_config.save_pretrained(args.output_dir)
    print("Generation configuration saved.")

    # 6. Evaluation
    torch.backends.cudnn.benchmark = True
    print("Evaluating baseline Falcon-7B...")
    baseline_rouge, baseline_bleu = evaluate_model(base_model_name, base_model_name, test_dataset, batch_size=16)
    print("\nBaseline Results:")
    print(f"ROUGE: {baseline_rouge}")
    print(f"BLEU: {baseline_bleu['bleu']}")

    print("\nEvaluating fine-tuned model...")
    finetuned_rouge, finetuned_bleu = evaluate_model(args.output_dir, args.output_dir, test_dataset, batch_size=16)
    print("\nFine-tuned Results:")
    print(f"ROUGE: {finetuned_rouge}")
    print(f"BLEU: {finetuned_bleu['bleu']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Falcon-7B on mental health counseling conversations.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./falcon-7b-instruct-mental-health-counseling",
        help="The directory where the model checkpoints and generation config will be saved."
    )
    # Add more command-line arguments here if needed
    args = parser.parse_args()
    main(args)
