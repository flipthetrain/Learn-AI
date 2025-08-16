import os
import json

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# ---------------------------------------------------------------------------
# Tool implementations (identical to the OpenAI version for comparison)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    mock_data = {
        "london":   {"temp_c": 12, "condition": "Cloudy",  "humidity": 78},
        "new york": {"temp_c": 18, "condition": "Sunny",   "humidity": 55},
        "tokyo":    {"temp_c": 22, "condition": "Partly cloudy", "humidity": 65},
    }
    data = mock_data.get(city.lower(), {"temp_c": 20, "condition": "Unknown", "humidity": 60})
    return {"city": city, **data}

def calculate(expression: str) -> dict:
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return {"error": "Expression contains disallowed characters"}
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}

def get_stock_price(ticker: str) -> dict:
    mock_prices = {"AAPL": 189.50, "GOOGL": 175.30, "MSFT": 415.20, "NVDA": 875.00}
    price = mock_prices.get(ticker.upper())
    if price is None:
        return {"ticker": ticker, "error": "Ticker not found"}
    return {"ticker": ticker.upper(), "price_usd": price}

TOOL_FUNCTIONS = {
    "get_weather":     get_weather,
    "calculate":       calculate,
    "get_stock_price": get_stock_price,
}

# ---------------------------------------------------------------------------
# Anthropic tool schemas (input_schema uses JSON Schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London'"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a numeric math expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(12 + 8) * 3'"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"}
            },
            "required": ["ticker"],
        },
    },
]

QUESTIONS = [
    "What's the weather like in Tokyo right now?",
    "If I buy 15 shares of NVDA at current price, what's my total cost?",
    "What is (2 ** 10) + (3 * 7)?",
]

try:
    import anthropic

    api_key = os.getenv('ANTHROPIC_API_KEY') or input('Enter your Anthropic API key: ')
    client = anthropic.Anthropic(api_key=api_key)

    results = []

    for question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"User: {question}")

        messages = [{"role": "user", "content": question}]

        # Agentic loop
        while True:
            response = client.messages.create(
                model='claude-haiku-4-5',
                max_tokens=512,
                tools=TOOLS,
                messages=messages,
            )

            # Append assistant turn (content is a list of blocks)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == 'end_turn':
                # Final text answer
                text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
                answer = ' '.join(text_blocks).strip()
                print(f"Assistant: {answer}")
                results.append({'question': question, 'answer': answer})
                break

            if response.stop_reason == 'tool_use':
                # Execute all tool_use blocks and collect results
                tool_results = []
                for block in response.content:
                    if block.type != 'tool_use':
                        continue
                    fn_name = block.name
                    fn_args = block.input
                    print(f"  [tool call] {fn_name}({fn_args})")

                    tool_result = TOOL_FUNCTIONS[fn_name](**fn_args)
                    print(f"  [tool result] {tool_result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result),
                    })

                messages.append({"role": "user", "content": tool_results})

    # Export to CSV
    import csv
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, os.path.splitext(os.path.basename(__file__))[0] + '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults exported to {csv_path}")

except Exception as e:
    print(f"An error occurred. See {exception_log} for details.")
    log_exception(e)
    raise
