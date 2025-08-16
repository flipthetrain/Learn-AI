# AI Agents and Tool Use

## What is an AI Agent?

An AI agent is an LLM that can take *actions* — not just generate text, but call external functions, APIs, or services, then use the results to continue reasoning and produce a final answer.

The model decides *when* to call a tool, *what arguments* to pass, and *how* to interpret the result.

---

## The Tool Use Loop

```
User Message
     │
     ▼
[LLM] ──► decides to call tool(s)
     │
     ▼
[Tool Execution] ──► results returned to LLM
     │
     ▼
[LLM] ──► incorporates results, may call more tools or respond to user
     │
     ▼
Final Response
```

This loop can repeat multiple times before producing a final answer.

---

## Why It Matters

Without tools, LLMs are limited to their training data. With tools they can:
- Look up live data (weather, stocks, databases)
- Perform precise computation (calculators, code execution)
- Take actions in the world (send emails, create calendar events)
- Chain multiple steps into complex workflows

---

## Example Tools in These Scripts

All examples use the same three tools:

| Tool | Description |
|---|---|
| `get_weather(city)` | Returns current conditions for a city |
| `calculate(expression)` | Evaluates a math expression safely |
| `get_stock_price(ticker)` | Returns a mock stock price |

Both the OpenAI and Anthropic scripts answer the same three questions
using these tools so you can compare how each API handles tool use.

---

## Python Examples

| Script | Provider | API Pattern |
|---|---|---|
| [agent_openai.py](./agent_openai.py) | OpenAI | `tools` + `tool_choice` in Chat Completions |
| [agent_anthropic.py](./agent_anthropic.py) | Anthropic | `tools` + `tool_use` content blocks |

---

## References

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [ReAct: Reasoning and Acting in Language Models (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)
