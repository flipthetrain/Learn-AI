import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# Anthropic's Voyage embedding models are accessed via the anthropic SDK.
# voyage-3 is their general-purpose embedding model (1024 dimensions).
# voyage-3-lite is a faster, smaller variant (512 dimensions).
# voyage-code-3 is optimised for code retrieval.

try:
    import anthropic

    api_key = os.getenv('ANTHROPIC_API_KEY') or input('Enter your Anthropic API key: ')
    client = anthropic.Anthropic(api_key=api_key)

    sentences = [
        "The dog is a loyal pet.",
        "That old car is a real dog.",
        "He called me a dog as an insult."
    ]

    vectors = []
    for sent in sentences:
        # client.beta.messages is for text; embeddings use a separate endpoint
        response = client.embeddings.create(
            model="voyage-3",
            input=[sent],
        )
        vector = np.array(response.data[0].embedding)
        vectors.append(vector)
        print(f"Sentence: {sent}\nEmbedding (first 8 dims): {vector[:8]}\n")

    # Export embeddings to CSV
    import csv
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, os.path.splitext(os.path.basename(__file__))[0] + '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence'] + [f'embedding_{i}' for i in range(len(vectors[0]))])
        for sent, vec in zip(sentences, vectors):
            writer.writerow([sent] + list(vec))
    print(f"Embeddings exported to {csv_path}")

    # Compare senses using cosine similarity
    if len(vectors) == 3:
        sim_pet_car    = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        sim_pet_insult = cosine_similarity([vectors[0]], [vectors[2]])[0][0]
        sim_car_insult = cosine_similarity([vectors[1]], [vectors[2]])[0][0]
        print("Cosine Similarity between senses of 'dog':")
        print(f"Pet vs Doesn't Work: {sim_pet_car:.4f}")
        print(f"Pet vs Insult:        {sim_pet_insult:.4f}")
        print(f"Doesn't Work vs Insult: {sim_car_insult:.4f}")

except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
