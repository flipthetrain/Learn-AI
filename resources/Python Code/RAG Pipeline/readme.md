# Retrieval-Augmented Generation (RAG)

## What is RAG?

Large language models have a knowledge cutoff — they only know what was in their training data. RAG solves this by giving the model access to an external knowledge base *at query time*.

The three steps:
1. **Index** — chunk your documents and embed each chunk into a vector
2. **Retrieve** — embed the user's question, find the most similar chunks
3. **Generate** — pass the retrieved chunks as context to an LLM and ask it to answer

This means the model can answer questions about documents it never saw during training.

---

## How It Works

```
User Question
     │
     ▼
[Embed Question]  ──────────────────────────────────────────────────────┐
                                                                        │
                                                                        ▼
Knowledge Base ──► [Embed Docs] ──► [Vector Store] ──► [Top-K Similar Chunks]
                                                                        │
                                                                        ▼
                              [LLM: "Given this context, answer the question"]
                                                                        │
                                                                        ▼
                                                                    Answer
```

---

## Why It Matters

- **Up-to-date**: inject fresh documents without retraining
- **Grounded**: the model cites specific passages instead of hallucinating
- **Cost-efficient**: cheaper than fine-tuning for factual knowledge tasks

---

## Toy Example: Cosine Similarity

The core of retrieval is comparing a query vector to stored document vectors.
For two vectors **a** and **b**:

```
similarity = (a · b) / (‖a‖ · ‖b‖)
```

Higher similarity = more relevant document.

---

## Python Examples

Each script builds a small AI-topic knowledge base, embeds it, then answers
three sample questions using RAG.

| Script | Embeddings | LLM |
|---|---|---|
| [rag_local.py](./rag_local.py) | sentence-transformers (local) | HuggingFace pipeline (local) |
| [rag_openai.py](./rag_openai.py) | text-embedding-3-small | gpt-4o-mini |
| [rag_anthropic.py](./rag_anthropic.py) | sentence-transformers (local) | claude-haiku-3-5 |
| [rag_ollama.py](./rag_ollama.py) | nomic-embed-text | llama3 |

All scripts export retrieved chunk similarity scores to `csv/`.

---

## References

- [Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](../../KeyPapers/RAG_Retrieval-Augmented_Generation_2020.pdf)
- [LangChain RAG documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone — What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
