import sys
import os

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

try:
    from transformers import BertTokenizer, BertModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Example sentences for each context
    sentences = [
        "The dog is a loyal pet.",  # pet animal
        "That old car is a real dog.",  # doesn't work well
        "He called me a dog as an insult."  # personal insult
    ]

    vectors = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the token index for 'dog'
        token_ids = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        dog_indices = [i for i, t in enumerate(tokens) if t == 'dog']
        for idx in dog_indices:
            vector = outputs.last_hidden_state[0, idx, :].numpy()
            vectors.append(vector)
            print(f"Sentence: {sent}\n'dog' vector (full): {vector}\n")

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
        sim_pet_car = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        sim_pet_insult = cosine_similarity([vectors[0]], [vectors[2]])[0][0]
        sim_car_insult = cosine_similarity([vectors[1]], [vectors[2]])[0][0]
        print("Cosine Similarity between senses of 'dog':")
        print(f"Pet vs Doesn't Work: {sim_pet_car:.4f}")
        print(f"Pet vs Insult: {sim_pet_insult:.4f}")
        print(f"Doesn't Work vs Insult: {sim_car_insult:.4f}")
except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
