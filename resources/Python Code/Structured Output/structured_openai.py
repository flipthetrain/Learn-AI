import os
import json

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# Reviews to analyse
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

# JSON Schema that the model must conform to
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the review"
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Inferred star rating from 1 (worst) to 5 (best)"
        },
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key topics or themes mentioned in the review"
        },
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the review"
        }
    },
    "required": ["sentiment", "rating", "themes", "summary"],
    "additionalProperties": False,
}

try:
    from openai import OpenAI

    api_key = os.getenv('OPENAI_API_KEY') or input('Enter your OpenAI API key: ')
    client = OpenAI(api_key=api_key)

    results = []

    for review in REVIEWS:
        print(f"\n{'='*60}")
        print(f"Review [{review['id']}]: {review['text'][:80]}...")

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": "You are a review analysis assistant. Analyse the review and respond with a JSON object."
                },
                {
                    "role": "user",
                    "content": f"Analyse this review:\n\n{review['text']}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "review_analysis",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                }
            },
        )

        analysis = json.loads(response.choices[0].message.content)
        print(f"  sentiment : {analysis['sentiment']}")
        print(f"  rating    : {analysis['rating']}/5")
        print(f"  themes    : {', '.join(analysis['themes'])}")
        print(f"  summary   : {analysis['summary']}")

        results.append({"review_id": review["id"], **analysis, "themes": ", ".join(analysis["themes"])})

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
