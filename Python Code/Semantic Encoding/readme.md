# Semantic Encoding: The Word "Dog" in Different Contexts

## Question

How would you encode the word "dog" when talking about:

- an animal that is a pet
- something that doesn't work well ("that car is a dog")
- as a personal insult ("he's such a dog")

---

## 1. Toy Vectorization Example

Suppose we use a simple 5-dimensional toy embedding space, where each dimension represents a semantic feature:

1. Animalness
2. Pet-likeness
3. Malfunction/Low Quality
4. Insult/Pejorative
5. General Frequency

| Context      | Vector Example            | Explanation                             |
| ------------ | ------------------------- | --------------------------------------- |
| Pet Animal   | [0.9, 0.8, 0.0, 0.0, 0.7] | High animal/pet, not malfunction/insult |
| Doesn't Work | [0.1, 0.0, 0.9, 0.1, 0.5] | High malfunction, low animal/pet        |
| Insult       | [0.2, 0.0, 0.1, 0.9, 0.4] | High insult, low animal/pet/malfunction |

---

## 2. Gemini 2.5 Pro Actual Vectors

> **Note:** Gemini 2.5 Pro (and similar LLMs) use high-dimensional embeddings (e.g., 768 or 1024+ dimensions). The actual vectors are not public, but here is a plausible illustration using 8 dimensions for demonstration. These are not the real Gemini vectors, but are formatted as if they were, to illustrate the concept.

### (A) "dog" as a pet animal

```
[0.12, 0.85, -0.03, 0.44, 0.01, 0.67, -0.12, 0.30]
```

### (B) "dog" meaning something that doesn't work well

```
[-0.22, 0.05, 0.91, -0.10, 0.33, -0.05, 0.12, 0.08]
```

### (C) "dog" as a personal insult

```
[0.05, -0.12, 0.08, 0.77, 0.60, -0.09, 0.02, 0.15]
```

---

## 3. Explanation

- The same word "dog" is mapped to different points in semantic space depending on context.
- Contextual encoders (like Gemini 2.5 Pro, GPT-4, BERT, etc.) generate different vectors for the same word in different sentences.
- The toy vectors show how features can be weighted for each meaning.
- The Gemini-style vectors are for illustration; real vectors are much higher-dimensional and proprietary.

---

## 4. Python Examples: Encoding "dog" in Context with BERT and LLM APIs

Below are Python scripts for encoding the word "dog" in different contexts and comparing their senses using cosine similarity. Each script is located in this folder:

- [BERT Example (local)](./bert_example.py)
- [OpenAI/ChatGPT API Example](./openai_example.py)
- [Ollama (Local LLMs) Example](./ollama_example.py)
- [Azure OpenAI Service Example](./azure_example.py)
- [AWS Bedrock (Amazon Titan Embeddings) Example](./aws_bedrock_example.py)

Each script demonstrates how to obtain contextual embeddings for "dog" in three different sentences and compare their semantic similarity.

---

## 5. References

- [Word Embeddings and Contextual Representations](https://jalammar.github.io/illustrated-bert/)
- [Google AI Blog: BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- [Gemini Model Card (Google)](https://deepmind.google/technologies/gemini/)
