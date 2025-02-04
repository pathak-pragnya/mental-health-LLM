# Falcon-7B Mental Health Counseling Fine-Tuning

This project fine-tunes the Falcon-7B model on a dataset of mental health counseling conversations. It leverages state-of-the-art techniques such as 4-bit quantization and LoRA (Low-Rank Adaptation) for efficient training. The project is modularized into several components including data loading/preprocessing, model building, training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Configuration](#configuration)

## Overview

This project fine-tunes the Falcon-7B-Instruct model to generate empathetic and contextually appropriate responses for mental health counseling using LoRA and quantization techniques.

**Features:**
- **Data Preprocessing:** Custom prompt templates and quality checks for context-response pairs.
- **Fine-Tuning:** Low-Rank Adaptation (LoRA) and 4-bit quantization for efficient training.
- **Evaluation:** Evaluation using ROUGE and BLEU metrics.

**Dataset:**
- **Name:** Mental Health Counseling Conversations Dataset
- **Description:** A curated collection of conversations between mental health counselors and clients.
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

**Technical Details:**
- **Model:** Falcon-7B-Instruct
- **Techniques:**
  - Low-Rank Adaptation (LoRA)
  - 4-bit Quantization (using BitsAndBytes)
- **Frameworks:** PyTorch, Hugging Face Transformers, BitsAndBytes, PEFT, TRL
- **Hardware:** Trained on Colab Pro (A100 GPUs)

**Results:**

**Baseline Model:**

- **ROUGE Scores:**
  - ROUGE-1: 0.1687
  - ROUGE-2: 0.0209
  - ROUGE-L: 0.1147
  - ROUGE-Lsum: 0.1066
- **BLEU Score:** 0.0069

**Fine-tuned Model:**

- **ROUGE Scores:**
  - ROUGE-1: 0.2164
  - ROUGE-2: 0.0341
  - ROUGE-L: 0.1208
  - ROUGE-Lsum: 0.1239
- **BLEU Score:** 0.0135

## Project Structure

```plaintext
my_project/
├── callbacks.py        # Custom training callbacks
├── data.py             # Data loading and preprocessing functions
├── evaluate.py         # Model evaluation code using ROUGE and BLEU metrics
├── main.py             # Entry point that ties all components together
├── model.py            # Model and tokenizer setup, including LoRA application
├── train.py            # Training loop and training arguments setup
├── requirements.txt    # List of dependencies
└── README.md           # This file


## Installation

```
bash

# 1. Clone the repository
git clone https://github.com/pathak-pragnya/mental-health-LLM
cd your-repo-name

# 2. Set Up a Virtual Environment (Recommended)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

## Usage

# Training the Model
1. Prepare the Datasets
The script automatically downloads the dataset from Hugging Face and applies preprocessing. No extra steps required unless you want to use a custom dataset.

2. Run the Main Script
From the root directory, run:
```
python main.py --output_dir ./falcon-7b-instruct-mental-health-counseling

  - The script will:
    1. Download and preprocess the dataset.
    2. Build and quantize the Falcon-7B model (4-bit).
    3. Apply LoRA adaptations.
    4. Train on the dataset.
    5. Save the fine-tuned model and generation config in the specified output_dir.

## Evaluating the Model

The main script also handles evaluation automatically. It computes ROUGE and BLEU metrics on the test dataset for both the baseline (Falcon-7B-Instruct) and the fine-tuned model. The results will be printed in your terminal at the end of the run.

If you only want to run the evaluation step with an already fine-tuned model, you can modify the code in main.py or create a separate script that loads the model from output_dir and calls the evaluation function in evaluate.py.

## Configuration

Most configuration settings are defined in the following places:

- train.py:
  get_training_args() specifies hyperparameters such as learning rate, batch size, number of epochs, etc.
- model.py:
  build_model_and_tokenizer() and apply_lora() define model loading, quantization settings, and LoRA parameters.
- main.py:
  Uses argparse to accept command-line arguments (e.g., --output_dir).
  You can customize these parameters directly in the functions or by extending argparse in main.py.