# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def build_model_and_tokenizer(base_model_name, bnb_config):
    """Loads and prepares the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        #quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    return model, tokenizer

def apply_lora(model):
    """Applies LoRA configuration to the model."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model
