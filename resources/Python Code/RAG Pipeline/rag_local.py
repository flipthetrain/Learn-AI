import os
import numpy as np

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# Small knowledge base about AI topics (the "documents" we will retrieve from)
DOCUMENTS = [
    "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. "
    "It replaced recurrent networks with self-attention mechanisms, enabling parallelisation and better long-range dependencies.",

    "BERT (Bidirectional Encoder Representations from Transformers) was released by Google in 2018. "
    "It pre-trains on masked language modelling and next-sentence prediction, producing rich contextual embeddings used for downstream tasks.",

    "GPT-3 is a 175-billion-parameter language model released by OpenAI in 2020. "
    "It demonstrated that very large models exhibit few-shot learning, performing new tasks from just a few examples in the prompt.",

    "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method introduced in 2021. "
    "Instead of updating all model weights, LoRA adds small trainable rank-decomposition matrices, reducing trainable parameters by up to 10,000x.",

    "Retrieval-Augmented Generation (RAG) was proposed by Lewis et al. in 2020. "
    "It combines a dense retrieval system with a generative model, allowing the model to access external knowledge at inference time.",

    "FlashAttention (2022) is an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU HBM and SRAM. "
    "It achieves 2-4x speedups over standard attention while using significantly less GPU memory.",

    "Chain-of-thought prompting (Wei et al. 2022) encourages LLMs to reason step-by-step before answering. "
    "Adding the phrase 'Let's think step by step' or providing reasoning examples significantly improves performance on arithmetic and logic tasks.",

    "Word2Vec (Mikolov et al. 2013) introduced efficient neural word embeddings trained with Skip-gram or CBOW objectives. "
    "The resulting vectors capture semantic relationships — the famous example: king - man + woman ≈ queen.",
]

QUESTIONS = [
    "What is the Transformer architecture and when was it introduced?",
    "How does LoRA reduce the cost of fine-tuning?",
    "What is RAG and why is it useful?",
]

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # --- INDEX: embed all documents ---
    print("Embedding knowledge base...")
    doc_embeddings = embedder.encode(DOCUMENTS)   # shape: (n_docs, dim)

    # Load a small generative model for the generation step
    print("Loading generation model (distilgpt2)...")
    generator = pipeline('text-generation', model='distilgpt2', max_new_tokens=80)

    results = []

    for question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Question: {question}")

        # --- RETRIEVE: find top-2 most relevant chunks ---
        q_embedding = embedder.encode([question])
        scores = cosine_similarity(q_embedding, doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:2]

        print("Retrieved chunks:")
        context_parts = []
        for rank, idx in enumerate(top_indices):
            print(f"  [{rank+1}] (score={scores[idx]:.4f}) {DOCUMENTS[idx][:80]}...")
            context_parts.append(DOCUMENTS[idx])
            results.append({
                'question': question,
                'rank': rank + 1,
                'score': scores[idx],
                'chunk': DOCUMENTS[idx],
            })

        # --- GENERATE: prompt the model with retrieved context ---
        context = " ".join(context_parts)
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        output = generator(prompt, do_sample=False)[0]['generated_text']
        # Extract only the answer portion
        answer = output[len(prompt):].strip().split('\n')[0]
        print(f"Answer: {answer}")

    # Export retrieval scores to CSV
    import csv
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, os.path.splitext(os.path.basename(__file__))[0] + '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'rank', 'score', 'chunk'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRetrieval scores exported to {csv_path}")

except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
    raise
