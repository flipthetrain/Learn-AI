# LoRA and QLoRA Fine-Tuning

## What is Fine-Tuning?

Pre-trained LLMs are general-purpose. Fine-tuning adapts them to a specific
task or style by continuing training on a smaller, targeted dataset.

The problem: updating all weights in a 7B+ parameter model requires enormous
GPU memory and time.

---

## LoRA: Low-Rank Adaptation

Instead of updating all original weights **W**, LoRA freezes **W** and adds
two small trainable matrices **A** and **B**:

```
W_new = W + A × B

where:
  W ∈ R^(d×k)   — frozen original weight
  A ∈ R^(d×r)   — trainable, r << d
  B ∈ R^(r×k)   — trainable
```

Because **r** (the "rank") is small (typically 4–64), the number of trainable
parameters drops by 10,000× or more. Only A and B are updated during training.

---

## QLoRA: Quantized LoRA

QLoRA combines LoRA with 4-bit quantization of the frozen base model weights,
reducing memory further. A 65B-parameter model that would need ~130 GB of
VRAM at full precision can be fine-tuned on a single 24 GB GPU with QLoRA.

```
Base Model (4-bit quantized, frozen)
     +
LoRA Adapters (fp16 or bf16, trainable)
     =
Fine-tuned behavior, fraction of the memory
```

---

## Example Task: Sentiment Classification

The script fine-tunes `distilgpt2` (small, fast) on a tiny sentiment dataset.
This is intentionally minimal — it demonstrates the mechanics of PEFT/LoRA
training without requiring a powerful GPU.

For real fine-tuning jobs, swap `distilgpt2` for a larger instruct model
(e.g. `meta-llama/Llama-3.2-1B-Instruct`) and use a proper dataset.

---

## Python Example

| Script | Description |
|---|---|
| [lora_finetune.py](./lora_finetune.py) | QLoRA fine-tuning with HuggingFace PEFT on distilgpt2 |

Requirements: `transformers`, `peft`, `datasets`, `torch`

---

## Key Hyperparameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `r` (rank) | 4–64 | Higher = more capacity, more memory |
| `lora_alpha` | 16–64 | Scales the LoRA updates (usually 2× rank) |
| `lora_dropout` | 0.0–0.1 | Regularization |
| `target_modules` | varies by model | Which weight matrices get LoRA adapters |

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)](../../KeyPapers/LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al. 2023)](../../KeyPapers/QLoRA_Efficient_Finetuning_of_Quantized_LLMs_2023.pdf)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
