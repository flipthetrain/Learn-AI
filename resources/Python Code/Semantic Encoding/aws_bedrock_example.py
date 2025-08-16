import boto3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

try:
    # Prompt for AWS credentials if not set
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    if not aws_access_key:
        aws_access_key = input('Enter your AWS Access Key ID: ')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    if not aws_secret_key:
        aws_secret_key = input('Enter your AWS Secret Access Key: ')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

    bedrock = boto3.client('bedrock-runtime',
        region_name=region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    sentences = [
        "The dog is a loyal pet.",
        "That old car is a real dog.",
        "He called me a dog as an insult."
    ]

    vectors = []
    for sent in sentences:
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=f'{{"inputText": "{sent}"}}',
            accept='application/json',
            contentType='application/json'
        )
        import json
        result = json.loads(response['body'].read())
        vector = np.array(result['embedding'])
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
except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
