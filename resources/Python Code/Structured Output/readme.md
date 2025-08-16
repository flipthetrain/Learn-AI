# Structured Output / JSON Mode

## What is Structured Output?

LLMs generate free-form text by default. Structured output forces the model to
respond in a specific format — typically JSON — that can be directly parsed and
used in application code without brittle string processing.

---

## Why It Matters

```
Without structured output:
  "The sentiment is positive and the rating is about 4 out of 5, with themes
   around great service and fast delivery."

With structured output:
  {
    "sentiment": "positive",
    "rating": 4,
    "themes": ["great service", "fast delivery"],
    "summary": "Customer had a positive experience with fast delivery."
  }
```

Structured output is essential for:
- Pipelines that feed LLM output into downstream systems
- Classification tasks with a fixed label set
- Extraction of named entities, key facts, or metadata
- Any time you need reliable, machine-readable responses

---

## Approaches

| Method | How It Works |
|---|---|
| **JSON mode** | Tell the model to output only valid JSON (no schema enforcement) |
| **Structured outputs / JSON schema** | Provide an explicit JSON Schema; model is constrained to match it |
| **Tool use trick** | Define a "save_result" tool with a typed schema; model must call it with valid args |
| **Format parameter** | Ollama / local models accept a `format: "json"` parameter |

---

## Example Task: Review Analysis

All scripts analyze the same product review and extract:

```json
{
  "sentiment": "positive" | "negative" | "neutral",
  "rating": 1-5,
  "themes": ["theme1", "theme2"],
  "summary": "one-sentence summary"
}
```

---

## Python Examples

| Script | Provider | Method |
|---|---|---|
| [structured_openai.py](./structured_openai.py) | OpenAI | JSON Schema (strict mode) |
| [structured_anthropic.py](./structured_anthropic.py) | Anthropic | Tool use pattern |
| [structured_ollama.py](./structured_ollama.py) | Ollama | `format: json` parameter |

All scripts export results to `csv/`.

---

## References

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Anthropic Tool Use for Structured Data](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [JSON Schema Specification](https://json-schema.org/)
