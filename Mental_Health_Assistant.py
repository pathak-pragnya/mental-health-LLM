import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys
import argparse
import evaluate
import nltk
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
nltk.download("punkt", quiet=True)

### Load the dataset
data = load_dataset("Amod/mental_health_counseling_conversations")

### Preprocess the dataset

#Create custom prompt
def create_prompt_template(context, response=None):
    prompt = (
        "Below is a conversation between a mental health counselor and a client.\n\n"
        "Client: {context}\n\n"
        "Counselor: "
    )

    if response:
        return prompt.format(context=context) + response
    return prompt.format(context=context)

#Clean the dataset : remove empty strings
def prepare_dataset(examples):
    prompts = []
    for context, response in zip(examples["Context"], examples["Response"]):
        if context.strip() and response.strip(): 
            prompts.append(create_prompt_template(context, response))
    return {"text": prompts}

### Callback

class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(f"Step {state.global_step}:")
            if "loss" in logs:
                print(f"  Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"  Eval Loss: {logs['eval_loss']:.4f}")
    def on_init_end(self, args, state, control, **kwargs):
        print("Training initialization complete.")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} complete.")    

def evaluate_model(model_name, tokenizer_name, dataset, batch_size=16):
    #Load the metrics
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    #Load the model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, padding_side="left")

    #Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    #Prepare the model for evaluation
    model = model.eval()

    #Torch inference optimizations
    with torch.inference_mode(), torch.cuda.amp.autocast():
        generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size, device=0)

        # Process texts in batches
        texts = [example["text"].split("Counselor:")[0].strip() for example in dataset]
        all_predictions = []
        all_references = []

        gen_text = [example["text"] for example in dataset]
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            generated = generation_pipeline(
                batch_texts,
                max_new_tokens=512,
                temperature=0.7,
                top_k=10,
                top_p=0.10,
                num_return_sequences=1,
                padding=True,
                truncation=True,
                repetition_penalty=1.1
            )
            batch_texts = gen_text[i:i + batch_size]

            for j, gen in enumerate(generated):
                if isinstance(gen, list):
                    generated_text = gen[0]["generated_text"]
                else:
                    generated_text = gen

                prediction = generated_text.split("Counselor:", 1)[-1].strip()
                if not prediction: 
                    prediction = generated_text
                
                reference = batch_texts[j].split("Counselor:")[-1].strip()
                if not reference:  
                    reference = batch_texts[j]
                # print("The question was: ", batch_texts[j].split("Counselor:")[0].strip())
                # print("The predicted text is: ", prediction)
                # print("The reference text is: ", reference)
                all_predictions.append(prediction)
                all_references.append(reference)
            # print(all_predictions)
            # print(all_references)
        
        rouge_scores = rouge_metric.compute(
            predictions=all_predictions,
            references=all_references,
            use_aggregator=True
        )

        tokenized_references = [[nltk.word_tokenize(ref.lower())] for ref in all_references]  
        tokenized_predictions = [nltk.word_tokenize(pred.lower()) for pred in all_predictions]  
        bleu_score = bleu_metric.compute(
            predictions=all_predictions,  
            references=[[ref] for ref in all_references] 
        )

        return rouge_scores, bleu_score
        
if __name__ == "__main__":
    #Train, test, validation split
    if {"train", "validation", "test"}.issubset(data.keys()):
        train_dataset = data["train"]
        val_dataset = data["validation"]
        test_dataset = data["test"]
    else:
        if "validation" not in data:
            split_data = data["train"].train_test_split(test_size=0.1, seed=42)
            train_dataset = split_data["train"]
            val_dataset = split_data["test"]
        else:
            train_dataset = data["train"]
            val_dataset = data["validation"]

        if "test" not in data:
            split_test = val_dataset.train_test_split(test_size=0.1, seed=42)
            val_dataset = split_test["train"]
            test_dataset = split_test["test"]
        else:
            test_dataset = data["test"]

    #apply the preprocessing function
    train_dataset = train_dataset.map( prepare_dataset, batched=True, remove_columns=train_dataset.column_names, num_proc=4)

    val_dataset = val_dataset.map( prepare_dataset, batched=True, remove_columns=val_dataset.column_names, num_proc=4)

    test_dataset = test_dataset.map(prepare_dataset, batched=True, remove_columns=test_dataset.column_names, num_proc=4)

     ### Loading the base model

    base_model_name = "tiiuae/falcon-7b-instruct"

    # # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    # # Load the base model tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # # Load the model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ### Prepare the model for kbit training
    model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # ### Training arguments
    output_dir = "./falcon-7b-instruct-mental-health-counseling"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 16
    optim = "adam"
    learning_rate = 5e-5
    warmup_ratio = 0.03
    weight_decay = 0.01
    seed = 42
    save_steps = 100
    logging_steps = 10
    max_steps = None
    max_train_samples = None
    max_val_samples = None
    max_test_samples = None
    fp16 = True
    group_by_length = True
    save_total_limit = 3
    evaluation_strategy = "steps"
    eval_steps = 10
    load_best_model_at_end = True
    gradient_checkpointing = True
    report_to = "none"
    dataloader_drop_last = False
    dataloader_num_workers = 4
    dataloader_pin_memory = True
    dataloader_timeout = None
    remove_unused_columns = True
    metric_for_best_model = "eval_loss"

    # ### LoRA configuration
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules = ["query_key_value", "dense", "dense", "dense_h_to_4h", "dense_4h_to_h"], lora_dropout=0.1, task_type="CAUSAL_LM", bias = "none")

    # ## Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # ### Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=fp16,
        group_by_length=group_by_length,
        save_total_limit=save_total_limit,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        dataloader_drop_last=dataloader_drop_last,
        dataloader_num_workers=dataloader_num_workers,
    )

    # ### Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[TrainingMonitorCallback()],
    )

    # ### Train the model
    trainer.train()

    # ### Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ### Evaluate the model
    results = trainer.evaluate()
    # print(results)

    # ### Config for inference
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

    generation_config.save_pretrained(output_dir)

    ### Load the model for inference
    torch.backends.cudnn.benchmark = True
    print("Evaluating baseline Falcon-7B...")
    baseline_rouge, baseline_bleu = evaluate_model("tiiuae/falcon-7b-instruct", "tiiuae/falcon-7b-instruct",test_dataset,batch_size=16)
    print("\nBaseline Results:")
    print(f"ROUGE: {baseline_rouge}")
    print(f"BLEU: {baseline_bleu['bleu']}")

    print("\nEvaluating fine-tuned model...")
    finetuned_rouge, finetuned_bleu = evaluate_model(output_dir, output_dir, test_dataset, batch_size=16)
    print("\nFine-tuned Results:")
    print(f"ROUGE: {finetuned_rouge}")
    print(f"BLEU: {finetuned_bleu['bleu']}")
