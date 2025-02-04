# train.py
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from callbacks import TrainingMonitorCallback

def get_training_args(output_dir, num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=16):
    """Defines training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_steps=100,
        logging_steps=10,
        fp16=True,
        group_by_length=True,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_drop_last=False,
        dataloader_num_workers=4,
    )

def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    """Initializes and runs the training process."""
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[TrainingMonitorCallback()],
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    return trainer
