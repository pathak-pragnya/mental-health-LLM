# data.py
import nltk
from datasets import load_dataset
from tqdm.auto import tqdm

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download("punkt", quiet=True)

def create_prompt_template(context, response=None):
    """Creates a conversation prompt from context and response."""
    prompt = (
        "Below is a conversation between a mental health counselor and a client.\n\n"
        "Client: {context}\n\n"
        "Counselor: "
    )
    if response:
        return prompt.format(context=context) + response
    return prompt.format(context=context)

def prepare_dataset(examples):
    """Preprocess the dataset by filtering and formatting examples."""
    prompts = []
    for context, response in zip(examples["Context"], examples["Response"]):
        if context.strip() and response.strip():
            prompts.append(create_prompt_template(context, response))
    return {"text": prompts}

def load_and_prepare_datasets():
    """Loads and splits the dataset, then applies preprocessing."""
    data = load_dataset("Amod/mental_health_counseling_conversations")

    # Handle different split cases
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

    # Apply preprocessing with multiple processes if available
    train_dataset = train_dataset.map(prepare_dataset, batched=True, remove_columns=train_dataset.column_names, num_proc=4)
    val_dataset = val_dataset.map(prepare_dataset, batched=True, remove_columns=val_dataset.column_names, num_proc=4)
    test_dataset = test_dataset.map(prepare_dataset, batched=True, remove_columns=test_dataset.column_names, num_proc=4)

    return train_dataset, val_dataset, test_dataset
