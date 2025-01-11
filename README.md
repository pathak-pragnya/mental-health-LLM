# Mental Health Assistant
Fine-tuning Falcon-7B-Instruct for empathetic mental health conversations

Overview:

This project fine-tunes the Falcon-7B-Instruct model to generate empathetic and contextually appropriate responses for mental health counseling using LoRA and quantization techniques.

Features:
  
  1. Data Preprocessing: Custom prompt templates and quality checks for context-response pairs.
  2. Fine-Tuning: Low-Rank Adaptation (LoRA) and 4-bit quantization for efficient training.
  3. Evaluation: Improved response quality, achieving significant gains in ROUGE and BLEU metrics.

Dataset:
  
  Name: Mental Health Counseling Conversations Dataset
  Description: A curated collection of conversations between mental health counselors and clients.
  Source: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

Technical Details
• Model: Falcon-7B-Instruct
• Techniques:
  -- Low-Rank Adaptation (LoRA)
  -- 4-bit Quantization
• Frameworks: PyTorch, Hugging Face Transformers, BitsAndBytes
• Hardware: Trained on Colab Pro (A100 GPUs)

Results:

Baseline Model:

The baseline evaluation of the Falcon-7B-instruct model produced the following results for ROUGE (Lin, 2004) and BLEU (Papineni et al., 2002) metrics:

• ROUGE Scores:
  
  – ROUGE-1: 0.1687
  
  – ROUGE-2: 0.0209
  
  – ROUGE-L: 0.1147
  
  – ROUGE-Lsum: 0.1066

• BLEU Score: 0.0069

Fine-tuned Model:

The results of the fine-tuned model are:

• ROUGE Scores:
  
  – ROUGE-1: 0.2164
  
  – ROUGE-2: 0.0341
  
  – ROUGE-L: 0.1208
  
  – ROUGE-Lsum: 0.1239

• BLEU Score: 0.0135
