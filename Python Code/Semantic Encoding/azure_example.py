import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

try:
    # Prompt for API key and endpoint if not set
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not AZURE_OPENAI_ENDPOINT:
        AZURE_OPENAI_ENDPOINT = input('Enter your Azure OpenAI endpoint URL: ')
    AZURE_API_KEY = os.getenv('AZURE_OPENAI_KEY')
    if not AZURE_API_KEY:
        AZURE_API_KEY = input('Enter your Azure OpenAI API key: ')

    headers = {
        'api-key': AZURE_API_KEY,
        'Content-Type': 'application/json'
    }

    sentences = [
        "The dog is a loyal pet.",
        "That old car is a real dog.",
        "He called me a dog as an insult."
    ]

    vectors = []
    for sent in sentences:
        data = {"input": sent}
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
        vector = np.array(response.json()['data'][0]['embedding'])
        vectors.append(vector)
        print(f"Sentence: {sent}\nEmbedding (first 8 dims): {vector[:8]}\n")

    # Cosine similarity as before
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
