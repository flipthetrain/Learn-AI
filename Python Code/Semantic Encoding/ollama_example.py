import requests
import numpy as np
import os
import sys

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

try:
    # Prompt for API URL, pre-populated with default
    default_url = 'http://localhost:11434/api/embeddings'
    api_url = input(f'Enter Ollama API URL [{default_url}]: ') or default_url

    sentences = [
        "The dog is a loyal pet.",
        "That old car is a real dog.",
        "He called me a dog as an insult."
    ]

    vectors = []
    for sent in sentences:
        response = requests.post(api_url, json={"model": "nomic-embed-text", "prompt": sent})
        vector = np.array(response.json()['embedding'])
        vectors.append(vector)
        print(f"Sentence: {sent}\nEmbedding (first 8 dims): {vector[:8]}\n")
except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
