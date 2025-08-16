import os
import json
import requests

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

REVIEWS = [
    {
        "id": "r1",
        "text": (
            "I've been using this laptop for three months now and I'm really impressed. "
            "The battery lasts all day, the keyboard is comfortable, and it boots in seconds. "
            "The only downside is the fan gets loud when I run heavy software. Overall a great buy."
        ),
    },
    {
        "id": "r2",
        "text": (
            "Terrible experience. The product arrived damaged and customer support took two weeks to respond. "
            "When they finally did, they offered a 10% discount instead of a replacement. Will not buy again."
        ),
    },
    {
        "id": "r3",
        "text": (
            "It's okay. Does what it says on the tin. Nothing spectacular but no major issues either. "
            "Shipping was fast, packaging was fine. Probably won't reorder but not unhappy."
        ),
    },
]

SYSTEM_PROMPT = """You are a review analysis assistant. When given a product review, respond with
ONLY a valid JSON object in this exact format — no other text:

{
  "sentiment": "positive" or "negative" or "neutral",
  "rating": integer 1-5,
  "themes": ["theme1", "theme2"],
  "summary": "one sentence summary"
}"""

try:
    default_base = 'http://localhost:11434'
    base_url = input(f'Enter Ollama base URL [{default_base}]: ').strip() or default_base
    model = input('Model [llama3]: ').strip() or 'llama3'

    results = []

    for review in REVIEWS:
        print(f"\n{'='*60}")
        print(f"Review [{review['id']}]: {review['text'][:80]}...")

        resp = requests.post(
            f'{base_url}/api/chat',
            json={
                "model": model,
                "stream": False,
                "format": "json",    # Ollama's built-in JSON mode
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Analyse this review:\n\n{review['text']}"}
                ],
            }
        )
        resp.raise_for_status()

        raw = resp.json()['message']['content']
        analysis = json.loads(raw)

        print(f"  sentiment : {analysis.get('sentiment')}")
        print(f"  rating    : {analysis.get('rating')}/5")
        themes = analysis.get('themes', [])
        print(f"  themes    : {', '.join(themes)}")
        print(f"  summary   : {analysis.get('summary')}")

        results.append({
            "review_id": review["id"],
            "sentiment": analysis.get("sentiment"),
            "rating":    analysis.get("rating"),
            "themes":    ", ".join(themes),
            "summary":   analysis.get("summary"),
        })

    # Export to CSV
    import csv
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, os.path.splitext(os.path.basename(__file__))[0] + '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['review_id', 'sentiment', 'rating', 'themes', 'summary'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults exported to {csv_path}")

except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
    raise
