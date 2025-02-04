# evaluate.py
import torch
import nltk
from tqdm.auto import tqdm
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def evaluate_model(model_name, tokenizer_name, dataset, batch_size=16):
    """Evaluates the model on a given dataset using ROUGE and BLEU metrics."""
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = model.eval()

    with torch.inference_mode(), torch.cuda.amp.autocast():
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=0
        )

        texts = [example["text"].split("Counselor:")[0].strip() for example in dataset]
        gen_text = [example["text"] for example in dataset]

        all_predictions = []
        all_references = []

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

            for j, gen in enumerate(generated):
                if isinstance(gen, list):
                    generated_text = gen[0]["generated_text"]
                else:
                    generated_text = gen

                prediction = generated_text.split("Counselor:", 1)[-1].strip()
                if not prediction: 
                    prediction = generated_text

                reference = gen_text[i+j].split("Counselor:")[-1].strip()
                if not reference:  
                    reference = gen_text[i+j]

                all_predictions.append(prediction)
                all_references.append(reference)

        rouge_scores = rouge_metric.compute(
            predictions=all_predictions,
            references=all_references,
            use_aggregator=True
        )

        bleu_score = bleu_metric.compute(
            predictions=all_predictions,
            references=[[ref] for ref in all_references]
        )

        return rouge_scores, bleu_score
