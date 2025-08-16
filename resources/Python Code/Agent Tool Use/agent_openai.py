import os
import json

exception_log = os.path.splitext(os.path.basename(__file__))[0] + '_exceptions.txt'

def log_exception(e):
    with open(exception_log, 'a') as f:
        f.write(str(e) + '\n')

# ---------------------------------------------------------------------------
# Tool implementations (the actual functions the model can invoke)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """Return mock weather data for a city."""
    mock_data = {
        "london":   {"temp_c": 12, "condition": "Cloudy",  "humidity": 78},
        "new york": {"temp_c": 18, "condition": "Sunny",   "humidity": 55},
        "tokyo":    {"temp_c": 22, "condition": "Partly cloudy", "humidity": 65},
    }
    data = mock_data.get(city.lower(), {"temp_c": 20, "condition": "Unknown", "humidity": 60})
    return {"city": city, **data}

def calculate(expression: str) -> dict:
    """Safely evaluate a numeric expression."""
    try:
        # Restrict to safe characters only
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return {"error": "Expression contains disallowed characters"}
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}

def get_stock_price(ticker: str) -> dict:
    """Return mock stock price for a ticker symbol."""
    mock_prices = {"AAPL": 189.50, "GOOGL": 175.30, "MSFT": 415.20, "NVDA": 875.00}
    price = mock_prices.get(ticker.upper())
    if price is None:
        return {"ticker": ticker, "error": "Ticker not found"}
    return {"ticker": ticker.upper(), "price_usd": price}

# Map tool names to Python functions
TOOL_FUNCTIONS = {
    "get_weather":     get_weather,
    "calculate":       calculate,
    "get_stock_price": get_stock_price,
}

# ---------------------------------------------------------------------------
# OpenAI tool schemas (JSON Schema format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'London'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric math expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(12 + 8) * 3'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"}
                },
                "required": ["ticker"],
            },
        },
    },
]

QUESTIONS = [
    "What's the weather like in Tokyo right now?",
    "If I buy 15 shares of NVDA at current price, what's my total cost?",
    "What is (2 ** 10) + (3 * 7)?",
]

try:
    from openai import OpenAI

    api_key = os.getenv('OPENAI_API_KEY') or input('Enter your OpenAI API key: ')
    client = OpenAI(api_key=api_key)

    results = []

    for question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"User: {question}")

        messages = [{"role": "user", "content": question}]

        # Agentic loop: keep going until no more tool calls
        while True:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                tools=TOOLS,
                tool_choice='auto',
            )
            msg = response.choices[0].message
            messages.append(msg)   # append assistant turn

            if not msg.tool_calls:
                # No more tool calls — final answer
                print(f"Assistant: {msg.content}")
                results.append({'question': question, 'answer': msg.content})
                break

            # Execute each tool call and feed results back
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"  [tool call] {fn_name}({fn_args})")

                tool_result = TOOL_FUNCTIONS[fn_name](**fn_args)
                print(f"  [tool result] {tool_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

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
