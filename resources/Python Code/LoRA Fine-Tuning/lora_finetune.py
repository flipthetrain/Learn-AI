import os
import json

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# ---------------------------------------------------------------------------
# Tiny sentiment dataset — 12 training examples, 4 validation
# In a real project, use datasets.load_dataset() with a proper corpus.
# ---------------------------------------------------------------------------
TRAIN_DATA = [
    {"text": "I love this product! It works perfectly.",         "label": "positive"},
    {"text": "Great quality, fast shipping, very happy.",        "label": "positive"},
    {"text": "Absolutely amazing, exceeded my expectations.",    "label": "positive"},
    {"text": "Best purchase I have made this year.",             "label": "positive"},
    {"text": "Wonderful experience from start to finish.",       "label": "positive"},
    {"text": "Five stars, highly recommend to everyone.",        "label": "positive"},
    {"text": "Terrible, broke after one day of use.",            "label": "negative"},
    {"text": "Worst product ever, complete waste of money.",     "label": "negative"},
    {"text": "Very disappointed, does not match description.",   "label": "negative"},
    {"text": "Poor quality, arrived damaged, avoid.",            "label": "negative"},
    {"text": "It is okay, nothing special about it.",            "label": "neutral"},
    {"text": "Average product, does the job but nothing more.",  "label": "neutral"},
]
EVAL_DATA = [
    {"text": "Really pleased with my order.",                   "label": "positive"},
    {"text": "Completely useless, returning immediately.",       "label": "negative"},
    {"text": "Decent enough for the price.",                    "label": "neutral"},
    {"text": "Outstanding! Will definitely buy again.",          "label": "positive"},
]

LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    MODEL_NAME = 'distilgpt2'    # Small model that runs on CPU; swap for a larger instruct model
    LORA_RANK = 4                # r — low-rank dimension
    LORA_ALPHA = 8               # scaling factor (typically 2 × rank)
    LORA_DROPOUT = 0.05

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Count original trainable parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_params:,}")

    # ---------------------------------------------------------------------------
    # Apply LoRA configuration
    # target_modules: which weight matrices to attach LoRA adapters to.
    # For GPT-2 style models these are the attention projection layers.
    # ---------------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn"],   # GPT-2 uses c_attn for Q/K/V projections
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable LoRA parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}% of total)")
    model.print_trainable_parameters()

    # ---------------------------------------------------------------------------
    # Prepare dataset
    # We frame sentiment as a text-completion task:
    #   "Review: <text> Sentiment: <label>"
    # The model learns to complete the prompt with the correct label.
    # ---------------------------------------------------------------------------
    def make_prompt(item):
        return f"Review: {item['text']} Sentiment: {item['label']}"

    def tokenize(batch):
        prompts = [make_prompt({"text": t, "label": l}) for t, l in zip(batch["text"], batch["label"])]
        enc = tokenizer(prompts, truncation=True, padding="max_length", max_length=64)
        enc["labels"] = enc["input_ids"].copy()   # causal LM: labels = input_ids
        return enc

    train_dataset = Dataset.from_list(TRAIN_DATA).map(tokenize, batched=True, remove_columns=["text", "label"])
    eval_dataset  = Dataset.from_list(EVAL_DATA).map(tokenize,  batched=True, remove_columns=["text", "label"])

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), 'lora_output')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=False,    # set True if you have a CUDA GPU with fp16 support
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()
    print("Training complete.")

    # ---------------------------------------------------------------------------
    # Simple inference: complete "Review: <text> Sentiment:" and read the next token
    # ---------------------------------------------------------------------------
    print("\n--- Inference on new examples ---")
    model.eval()
    test_texts = [
        "The packaging was lovely but the item itself is flimsy.",
        "Incredible value for money, I am thrilled with this.",
        "Stopped working after a week, very frustrating.",
    ]

    results = []
    for text in test_texts:
        prompt = f"Review: {text} Sentiment:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        predicted_label = generated.split()[0].lower() if generated else "unknown"
        print(f"  Input: {text[:60]}")
        print(f"  Predicted: {predicted_label}\n")
        results.append({"text": text, "predicted": predicted_label})

    # Save LoRA adapter weights
    adapter_path = os.path.join(output_dir, 'adapter')
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
    print("To reload: model = PeftModel.from_pretrained(base_model, adapter_path)")

    # Export results to CSV
    import csv
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, os.path.splitext(os.path.basename(__file__))[0] + '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'predicted'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Inference results exported to {csv_path}")

except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
    raise
