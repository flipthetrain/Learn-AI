# Learn AI

This document provides a curated list of resources for learning Artificial Intelligence (AI) and Large Language Models (LLMs), organized by academic level and topic.

## Table of Contents

- [Recommended Learning Order](#recommended-learning-order)
- [High School Level](#high-school-level)
  - [Introduction to Programming](#introduction-to-programming)
  - [Introduction to AI Concepts](#introduction-to-ai-concepts)
  - [Mathematics Foundations](#mathematics-foundations)
  - [Computer Science Basics](#computer-science-basics)
- [Undergraduate Level](#undergraduate-level)
  - [Essential Mathematics](#essential-mathematics)
  - [Essential Python](#essential-python)
  - [Foundational AI & Deep Learning](#foundational-ai--deep-learning)
  - [Foundational Papers](#foundational-papers)
- [Practitioner Level (2.5)](#practitioner-level-25)
  - [Prompt Engineering & API Usage](#prompt-engineering--api-usage)
  - [Model Selection & Cost Guidance](#model-selection--cost-guidance)
  - [RAG & Knowledge Systems](#rag--knowledge-systems)
  - [Building LLM Applications](#building-llm-applications)
- [Graduate Level](#graduate-level)
  - [Advanced Language Models](#advanced-language-models)
  - [Key Methods & Optimizations](#key-methods--optimizations)
  - [Multimodal Models](#multimodal-models)
  - [AI Agents](#ai-agents)
  - [AI Ethics & Safety](#ai-ethics--safety)
  - [AI Tools & Frameworks](#ai-tools--frameworks)
- [Doctoral Level & Research](#doctoral-level--research)
  - [Surveys and Overviews](#surveys-and-overviews)
  - [Specialized Topics](#specialized-topics)
- [Key Contributors & Resources](#key-contributors--resources)
  - [Key Personalities & Labs](#key-personalities--labs)
  - [YouTube Channels](#youtube-channels)
- [How to Contribute](#how-to-contribute)
- [Glossary](#glossary)
- [License](#license)

## Recommended Learning Order

A structured path for learning AI, especially focusing on LLMs:

1.  **Foundations (Undergraduate Level)**:
    *   **Mathematics**: Solidify your understanding of Linear Algebra, Calculus, Probability, and Statistics.
    *   **Python**: Become proficient in Python and its scientific computing libraries (NumPy, PyTorch).
    *   **Deep Learning Basics**: Learn about neural networks, backpropagation, and fundamental architectures. Watch Andrej Karpathy's "Neural Networks from Scratch".
    *   **Transformers**: Understand the Transformer architecture, which is the foundation of modern LLMs. Read "Attention Is All You Need".

1.5. **Production Skills (Practitioner Level)**:
    *   **Prompt Engineering**: Learn systematic techniques for getting reliable, high-quality outputs from LLMs.
    *   **Model Selection**: Understand the cost/capability tradeoffs between frontier APIs (GPT-4o, Claude, Gemini) and local open-weight models (LLaMA, Mistral, Phi).
    *   **RAG**: Build retrieval-augmented systems that give models access to your own documents and data.
    *   **LLM Application Frameworks**: Get hands-on with LangChain, LlamaIndex, and Ollama to ship real applications quickly.
    *   **Agents & Tool Use**: Learn how LLMs can call external tools, browse the web, and complete multi-step tasks autonomously.

2.  **Core LLM Concepts (Graduate Level)**:
    *   **GPT Series**: Read the foundational GPT papers (GPT-1, GPT-2, GPT-3) to understand the evolution of generative pre-trained models.
    *   **Fine-tuning & RLHF**: Learn about techniques like fine-tuning, Reinforcement Learning from Human Feedback (RLHF), and parameter-efficient methods (LoRA).
    *   **Multimodal Models**: Understand how vision encoders are connected to LLMs, and how to use image inputs via the OpenAI and Anthropic APIs.
    *   **LLM Development**: Watch courses on building and developing LLMs to translate theory into practice.

3.  **Advanced Topics & Research (Doctoral Level)**:
    *   **Efficiency**: Study methods for making LLMs more efficient (e.g., FlashAttention, DistilBERT).
    *   **Safety & Ethics**: Explore research on AI alignment, constitutional AI, and ethical considerations.
    *   **Surveys**: Read recent survey papers to get a broad overview of the state-of-the-art.

---

## High School Level

This section provides introductory resources for high school students interested in AI and computer science.

### Introduction to Programming

*   **Books:**
    *   **Think Python: How to Think Like a Computer Scientist (Downey)** - [🌐 Online](https://greenteapress.com/wp/think-python-2e/): Free, beginner-friendly introduction to programming with Python.
*   **Videos:**
    *   [Python for Beginners - Programming with Mosh](https://www.youtube.com/watch?v=_uQrJ0TkZlc): Complete beginner-friendly Python tutorial (6 hours).
    *   [CS50: Introduction to Computer Science - Harvard](https://www.youtube.com/playlist?list=PLhQjrBD2T380F_inVRXMIHCqLaNUd7bN4): World-famous introduction to computer science and programming.
    *   [Code.org - Python Course](https://code.org/educate/curriculum/high-school): Interactive high school programming curriculum.

### Introduction to AI Concepts

*   **Videos & Courses:**
    *   [Elements of AI - University of Helsinki](https://www.elementsofai.com/): Free online course combining theory with practical exercises, suitable for high school students.
    *   [MIT OpenCourseWare - Artificial Intelligence](https://ocw.mit.edu/search/?d=Electrical%20Engineering%20and%20Computer%20Science&t=Artificial%20Intelligence): Comprehensive AI courses from MIT.
    *   [AI for Everyone - Andrew Ng (Coursera)](https://www.coursera.org/learn/ai-for-everyone): Non-technical introduction to AI fundamentals.
    *   [CrashCourse - Artificial Intelligence](https://www.youtube.com/playlist?list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b): Engaging video series explaining AI concepts.
    *   [How AI Works - minutephysics](https://www.youtube.com/watch?v=R9OHn5ZF4Uo): Simple visual explanations of neural networks and machine learning.

### Mathematics Foundations

*   **Videos:**
    *   [Algebra I & II - Khan Academy](https://www.khanacademy.org/math/algebra): Essential algebraic foundations for AI.
    *   [Statistics & Probability - Khan Academy](https://www.khanacademy.org/math/statistics-probability): Statistical concepts needed for understanding AI.
    *   [Precalculus - Khan Academy](https://www.khanacademy.org/math/precalculus): Functions and mathematical thinking.

### Computer Science Basics

*   **Books:**
    *   **Computer Science Illuminated (Dale & Lewis)**: Comprehensive introduction to computer science principles.
*   **Videos:**
    *   [Crash Course Computer Science](https://www.youtube.com/playlist?list=PL8dPuuaLjXtNlUrzyH5r6jN9ulIgZBpdo): Comprehensive overview of computer science history and fundamentals.
    *   [How Computers Work - Code.org](https://www.youtube.com/playlist?list=PLzdnOPI1iJNcsRwJhvksEo1tJqjIqWbN-): Understanding hardware and software basics.

---

## Undergraduate Level

### Essential Mathematics
*   **Videos:**
    *   [Linear Algebra Fundamentals — 3Blue1Brown](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    *   [Calculus Essentials — 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
    *   [Statistics and Probability — Khan Academy](https://www.youtube.com/watch?v=uhxtUt_-GyM&list=PL1328115D3D8A2566)
    *   [Backpropagation Mathematics — 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### Essential Python
*   **Videos:**
    *   [Python Full Course for Beginners [2025] — FreeCodeCamp](https://www.youtube.com/watch?v=K5KVEU3aaeQ)
    *   [Python NumPy Tutorial — freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI)
    *   [PyTorch for Deep Learning — freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA)

### Foundational AI & Deep Learning
*   **Videos:**
    *   [Neural Networks from Scratch — Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
    *   [Deep Learning Fundamentals — DeepLearning.AI](https://www.youtube.com/watch?v=pyqt7s2bpqM)
    *   [How Transformers Work — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
    *   [The Transformer Architecture — Stanford CS224N](https://www.youtube.com/watch?v=S27pHKBEp30)

### Foundational Papers
*   **[Efficient Estimation of Word Representations in Vector Space (2013) by Mikolov et al.](./resources/KeyPapers/Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf)** ([External](https://arxiv.org/abs/1301.3781)) — Word2Vec; efficient word embeddings.
*   **[Attention Is All You Need (2017) by Vaswani et al.](./resources/KeyPapers/Attention_Is_All_You_Need_2017.pdf)** ([External](https://arxiv.org/abs/1706.03762)) — Introduced the Transformer architecture.
*   **[BERT: Pre-training of Deep Bidirectional Transformers (2018) by Devlin et al.](./resources/KeyPapers/BERT_2018.pdf)** ([External](https://arxiv.org/abs/1810.04805)) — Bidirectional pre-training for representation learning.
*   **[Improving Language Understanding by Generative Pre-Training (2018) by Radford et al.](./resources/KeyPapers/Improving_Language_Understanding_by_Generative_Pre-Training_2018.pdf)** ([External](https://openai.com/research/language-unsupervised)) — Early GPT demonstration of generative pre-training.
*   **[Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](./resources/KeyPapers/Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf)** ([External](https://openai.com/research/better-language-models)) — Scaling LMs for generalization.
*   **[Language Models are Few-Shot Learners (GPT-3, 2020) by Brown et al.](./resources/KeyPapers/GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf)** ([External](https://arxiv.org/abs/2005.14165)) — Few-shot capabilities emerge at scale.

---

## Practitioner Level (2.5)

For people who want to **build and ship real LLM applications** without going through the full graduate curriculum. Prerequisites: basic Python (Undergraduate Level), familiarity with what transformers are. No deep math required.

### Prompt Engineering & API Usage
*   **Courses & Guides:**
    *   [ChatGPT Prompt Engineering for Developers — DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/): Free short course covering zero-shot, few-shot, chain-of-thought, and system prompts.
    *   [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview): Comprehensive techniques specific to Claude models.
    *   [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering): Official OpenAI best practices with concrete examples.
*   **Key Techniques:**
    *   **Zero-shot prompting** — ask directly with no examples.
    *   **Few-shot prompting** — provide 2–5 input/output examples to steer behaviour.
    *   **Chain-of-thought** — add "think step by step" to improve reasoning on complex tasks.
    *   **System prompts** — set persona, output format, and constraints at the conversation level.
    *   **Structured output** — use JSON Schema or tool-use to get reliably parseable responses. See [Structured Output examples](./resources/Python%20Code/Structured%20Output/).

### Model Selection & Cost Guidance

Choosing the right model for a task is a key production skill. The landscape divides into two categories:

#### Frontier API Models (hosted, no GPU required)

| Provider | Model | Best For | Approx. Cost (input/output per 1M tokens) |
|---|---|---|---|
| Anthropic | claude-opus-4 | Complex reasoning, long documents, nuanced writing | $$$ |
| Anthropic | claude-sonnet-4 | Balanced intelligence and speed; most tasks | $$ |
| Anthropic | claude-haiku-4-5 | High-volume, latency-sensitive, simple tasks | $ |
| OpenAI | gpt-4o | Multimodal (text + images), strong reasoning | $$ |
| OpenAI | gpt-4o-mini | Fast, cheap; classification and extraction | $ |
| Google | gemini-2.5-pro | Long context (1M tokens), code, reasoning | $$ |

**When to use a frontier API:**
- You need the best possible quality right now
- You don't have a GPU or don't want to manage infrastructure
- Your use case involves < a few million tokens/day (cost is still manageable)
- You need multimodal input (images, audio, video)

#### Open-Weight Local Models (self-hosted)

| Model Family | Sizes | Best For |
|---|---|---|
| Meta LLaMA 3.x | 8B, 70B | General purpose; strong instruct-tuned variants |
| Mistral / Mixtral | 7B, 8×7B MoE | Fast, efficient; Mixtral matches GPT-3.5 at lower cost |
| Microsoft Phi-3/4 | 3.8B, 14B | Strong performance at very small sizes; edge deployment |
| Google Gemma 3 | 2B, 9B, 27B | Open, permissive licence; good for fine-tuning |
| Qwen 2.5 | 0.5B–72B | Multilingual; strong coding models in the family |

**When to use a local model:**
- Data privacy / compliance — data must not leave your network
- High volume — inference costs at scale make self-hosting cheaper
- Customisation — you plan to fine-tune on proprietary data
- Offline / edge deployment — no internet connection available

**Running local models with Ollama:**
```bash
# Install: https://ollama.com
ollama pull llama3          # download the model
ollama run llama3           # interactive chat
ollama serve                # start local API at http://localhost:11434
```
Ollama exposes a REST API compatible with the OpenAI client library, making it easy to swap between local and cloud models in code.

*   **Resources:**
    *   [Ollama — Run LLMs Locally](https://ollama.com): Easiest way to run open-weight models on your laptop.
    *   [LMStudio — GUI for Local LLMs](https://lmstudio.ai): Desktop app for downloading and chatting with local models, no CLI needed.
    *   [Artificial Analysis LLM Benchmarks](https://artificialanalysis.ai): Independent speed, quality, and cost comparisons across hosted models.

### RAG & Knowledge Systems

RAG (Retrieval-Augmented Generation) lets a model answer questions about documents it never saw during training. It is the most widely used pattern for grounding LLM outputs in your own data.

*   **Videos:**
    *   [RAG from Scratch — LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8): 15-part series building RAG incrementally from first principles.
    *   [Building RAG Applications — DeepLearning.AI](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/): Short course covering naive RAG, advanced retrieval, and evaluation.
*   **Vector Databases** (where embeddings are stored):
    *   [Chroma](https://www.trychroma.com) — lightweight, runs in-process, great for prototyping.
    *   [Qdrant](https://qdrant.tech) — production-grade, self-hostable or cloud.
    *   [Pinecone](https://www.pinecone.io) — managed cloud service, minimal ops overhead.
    *   [pgvector](https://github.com/pgvector/pgvector) — vector search inside PostgreSQL; good if you already use Postgres.
*   **Paper:**
    *   **[RAG: Retrieval-Augmented Generation (2020)](./resources/KeyPapers/RAG_Retrieval-Augmented_Generation_2020.pdf)** — the original Lewis et al. paper.
*   **Code Examples:** [RAG Pipeline](./resources/Python%20Code/RAG%20Pipeline/) — local, OpenAI, Anthropic, and Ollama implementations.

### Building LLM Applications
*   **Frameworks:**
    *   **[LangChain](https://python.langchain.com)** — composable chains and agents; large ecosystem of integrations. Best for quickly assembling pipelines from pre-built components.
    *   **[LlamaIndex](https://www.llamaindex.ai)** — focused on data ingestion, indexing, and retrieval. Excels at connecting LLMs to structured and unstructured data sources (PDFs, databases, APIs). Strong RAG primitives.
    *   **[LangGraph](https://langchain-ai.github.io/langgraph/)** — extends LangChain with stateful, graph-based agent workflows. Use when you need branching logic, loops, human-in-the-loop steps, or multi-agent coordination.
    *   **[Ollama](https://ollama.com)** — run open-weight models locally with a simple CLI and OpenAI-compatible REST API. Ideal for privacy-sensitive or offline use cases.
    *   **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** — the standard library for loading, fine-tuning, and running open-weight models in Python.
*   **Videos:**
    *   [LangChain for LLM Application Development — DeepLearning.AI](https://www.youtube.com/watch?v=_v_fgW2SkkQ)
    *   [LlamaIndex — Full Crash Course](https://www.youtube.com/watch?v=cNMYeW2mpBs)
    *   [LangGraph: Build Stateful AI Agents — LangChain](https://www.youtube.com/watch?v=R8KB-Zcynxc)
    *   [Ollama Tutorial — Run LLMs Locally](https://www.youtube.com/watch?v=1IQGpRZfxjk)
*   **Code Examples:**
    *   [Agent Tool Use](./resources/Python%20Code/Agent%20Tool%20Use/) — OpenAI and Anthropic function calling.
    *   [Structured Output](./resources/Python%20Code/Structured%20Output/) — JSON extraction with OpenAI, Anthropic, and Ollama.

---

## Graduate Level

### Advanced Language Models
*   **Papers:**
    *   **[Transformer-XL (2019) by Dai et al.](./resources/KeyPapers/Transformer-XL_2019.pdf)** ([External](https://arxiv.org/abs/1901.02860)) — Long-context modeling with recurrence and relative positions.
    *   **[T5: Text-to-Text Transfer Transformer (2020) by Raffel et al.](./resources/KeyPapers/Exploring_the_Limits_of_Transfer_Learning_with_a_Unified_Text-to-Text_Transformer_2020.pdf)** ([External](https://arxiv.org/abs/1910.10683)) — Unified text-to-text framework.
    *   **[ELECTRA (2020) by Clark et al.](./resources/KeyPapers/ELECTRA_2020.pdf)** ([External](https://arxiv.org/abs/2003.10555)) — Sample-efficient pre-training via replaced-token detection.
*   **Videos:**
    *   [Stanford CS229: Building Large Language Models — Stanford Online](https://www.youtube.com/watch?v=9vM4p9NN0Ts)
    *   [Developing Large Language Models in Python — NeuralNine](https://www.youtube.com/watch?v=s5nq-a1wpPY)
    *   [Create a Large Language Model from Scratch — FreeCodeCamp](https://www.youtube.com/watch?v=UU1WVnMk4E8)

### Key Methods & Optimizations
*   **Papers:**
    *   **[Training language models to follow instructions with human feedback (2022)](./resources/KeyPapers/Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf)** ([External](https://arxiv.org/abs/2203.02155)) — RLHF for alignment and helpfulness.
    *   **[LoRA: Low-Rank Adaptation of Large Language Models (2021)](./resources/KeyPapers/LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf)** ([External](https://arxiv.org/abs/2106.09685)) — Parameter-efficient fine-tuning.
    *   **[QLoRA: Efficient Finetuning of Quantized LLMs (2023) by Dettmers et al.](./resources/KeyPapers/QLoRA_Efficient_Finetuning_of_Quantized_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2305.14314)) — 4-bit quantization + LoRA; fine-tune 65B models on a single GPU.
    *   **[FlashAttention (2022) by Dao et al.](./resources/KeyPapers/FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf)** ([External](https://arxiv.org/abs/2205.14135)) — Fast, memory-efficient attention implementation.
    *   **[Chain-of-Thought Prompting (2022) by Wei et al.](./resources/KeyPapers/Chain-of-Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models_2022.pdf)** ([External](https://arxiv.org/abs/2201.11903)) — Improves reasoning via stepwise prompting.
    *   **[Direct Preference Optimization (2023) by Rafailov et al.](./resources/KeyPapers/DPO_Direct_Preference_Optimization_2023.pdf)** ([External](https://arxiv.org/abs/2305.18290)) — Simpler alternative to RLHF; optimises preferences directly without a reward model.
    *   **[RAG: Retrieval-Augmented Generation (2020) by Lewis et al.](./resources/KeyPapers/RAG_Retrieval-Augmented_Generation_2020.pdf)** ([External](https://arxiv.org/abs/2005.11401)) — Combines dense retrieval with generation for knowledge-intensive tasks.
*   **Videos:**
    *   [Fine-tuning LLMs w/ Example Code — Shawhin Talebi](https://www.youtube.com/watch?v=eC6Hd1hFvos)
    *   [RLHF: Training Language Models with Human Feedback — Hugging Face](https://www.youtube.com/watch?v=2MBJOuVq380)
    *   [RAG from Scratch — LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8)

### Multimodal Models

Multimodal models process more than one type of input — typically text combined with images, audio, or video. Since 2023 this has become a mainstream capability of frontier models rather than a research specialty.

**How vision-language models work:**

Early approaches (CLIP, Flamingo) aligned a separate vision encoder with a language model via a learned projection. Modern models (GPT-4o, Claude 3.x, Gemini) go further, natively interleaving image tokens with text tokens in a unified architecture, enabling richer reasoning over mixed-modal inputs.

```
Image → [Vision Encoder] → image tokens ─┐
                                          ├─► [LLM] → text response
Text  → [Tokenizer]      → text tokens  ─┘
```

**What frontier models support (as of 2025):**

| Model | Images | Video | Audio | Documents |
|---|---|---|---|---|
| GPT-4o (OpenAI) | Yes | Yes | Yes | Yes |
| Claude 3.5/3.7 (Anthropic) | Yes | No | No | Yes (via vision) |
| Gemini 2.x (Google) | Yes | Yes | Yes | Yes |
| LLaVA / LLaVA-Next (open) | Yes | No | No | No |
| Llama 3.2 Vision (Meta) | Yes | No | No | No |

**Common use cases:**
- Document understanding (invoices, forms, charts, slide decks)
- Image captioning and visual Q&A
- Screenshot-to-code
- Medical imaging analysis
- Video summarisation

*   **Key Papers:**
    *   **[CLIP: Learning Transferable Visual Models from Natural Language Supervision (2021) by Radford et al.](https://arxiv.org/abs/2103.00020)** — Contrastive pretraining to align image and text embeddings; the foundation of most vision-language work.
    *   **[Flamingo: A Visual Language Model for Few-Shot Learning (2022) by Alayrac et al.](https://arxiv.org/abs/2204.14198)** — DeepMind's approach to interleaving visual features with a frozen LLM for few-shot multimodal tasks.
    *   **[LLaVA: Large Language and Vision Assistant (2023) by Liu et al.](https://arxiv.org/abs/2304.08485)** — Instruction-tuned open-source vision-language model; approachable architecture for understanding how vision is connected to an LLM.
    *   **[GPT-4 Technical Report (2023)](./resources/KeyPapers/GPT-4_Technical_Report_2023.pdf)** ([External](https://arxiv.org/abs/2303.08774)) — Covers GPT-4V multimodal capabilities alongside language benchmarks.
*   **Videos:**
    *   [How GPT-4 Vision Works — Andrej Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A)
    *   [Multimodal LLMs Explained — Umar Jamil](https://www.youtube.com/watch?v=vAmKB7iPkWw)
    *   [Build a Multimodal AI App — FreeCodeCamp](https://www.youtube.com/watch?v=dXxQ0LR-3Hg)
*   **Practical API usage:**

    ```python
    # OpenAI — pass an image URL or base64-encoded image
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            ],
        }],
    )

    # Anthropic — same pattern with base64 source
    import anthropic, base64
    client = anthropic.Anthropic()
    with open("image.jpg", "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",  "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                {"type": "text",   "text": "What is in this image?"},
            ],
        }],
    )
    ```

### AI Agents

AI agents are LLMs that can take actions — calling tools, browsing the web, writing and executing code, and coordinating with other agents — to complete multi-step goals autonomously. This is now a core topic in applied AI, not just a research curiosity.

**The agent loop:**
1. Model receives a goal from the user
2. Model decides what tool(s) to call and with what arguments
3. Tool results are fed back to the model
4. Model reasons over results, calls more tools or produces a final answer

*   **Papers:**
    *   **[ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629)** — Introduced the Reason + Act loop that underpins most agent frameworks.
    *   **[Toolformer (2023) by Schick et al.](https://arxiv.org/abs/2302.04761)** — Training LLMs to teach themselves which APIs to call and when.
*   **Videos:**
    *   [Building AI Agents with Tool Use — Anthropic](https://www.youtube.com/watch?v=pMpBRFClMxE)
    *   [LangGraph: Build Stateful AI Agents — LangChain](https://www.youtube.com/watch?v=R8KB-Zcynxc)
    *   [Agentic AI Design Patterns — DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)
*   **Frameworks:**
    *   **[LangGraph](https://langchain-ai.github.io/langgraph/)** — graph-based stateful agents with branching, loops, and human-in-the-loop support.
    *   **[AutoGen (Microsoft)](https://microsoft.github.io/autogen/)** — multi-agent conversation framework where agents collaborate, critique, and verify each other.
    *   **[CrewAI](https://www.crewai.com)** — role-based multi-agent teams; good for structured pipelines (researcher → writer → reviewer).
    *   **[Claude Agent SDK / Anthropic SDK](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** — native tool use with Claude models; clean API for single-agent patterns.
*   **Code Examples:** [Agent Tool Use](./resources/Python%20Code/Agent%20Tool%20Use/) — OpenAI function calling and Anthropic tool use, side by side.

### AI Ethics & Safety
*   **Papers:**
    *   **[Constitutional AI: Harmlessness from AI Feedback (2022)](./resources/KeyPapers/Constitutional_AI_Harmlessness_from_AI_Feedback_2022.pdf)** ([External](https://arxiv.org/abs/2212.08073)) — Anthropic's method for training helpful, harmless models using AI-generated feedback.
    *   **[Collective Constitutional AI: Aligning a Language Model with Public Input](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)** — Extending Constitutional AI with democratically-sourced values.
*   **Videos:**
    *   [AI Alignment and Safety — Anthropic](https://www.youtube.com/watch?v=FX0zYxm4yes)
    *   [Ethics in AI Development — DeepMind](https://www.youtube.com/watch?v=z7tRnlqybnU)

### AI Tools & Frameworks

This section covers the production tooling layer — the libraries and platforms used to build, deploy, and evaluate LLM applications at scale.

*   **Orchestration & Application Frameworks:**
    *   **[LangChain](https://python.langchain.com)** — composable chains, retrieval, agents, and a large integration ecosystem. The most widely-used LLM application framework.
    *   **[LlamaIndex](https://www.llamaindex.ai)** — specialises in data ingestion, indexing, and retrieval. Stronger than LangChain for connecting LLMs to complex data sources (PDFs, SQL, APIs, knowledge graphs). Includes LlamaParse for structured document extraction.
    *   **[LangGraph](https://langchain-ai.github.io/langgraph/)** — stateful, graph-based agent workflows built on top of LangChain. Use it when your agents need branching, retry logic, multi-step planning, or human-in-the-loop interrupts.
*   **Local Model Serving:**
    *   **[Ollama](https://ollama.com)** — run open-weight models (LLaMA, Mistral, Phi, Gemma, Qwen, etc.) locally with a single command. Exposes an OpenAI-compatible REST API so you can swap cloud/local with one line of code. Supports GPU and CPU inference on Mac, Windows, and Linux.
    *   **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput inference server for production deployments; supports continuous batching and paged attention.
    *   **[LMStudio](https://lmstudio.ai)** — GUI desktop app for downloading, managing, and chatting with local models; no CLI required.
*   **Model Training & Fine-tuning:**
    *   **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** — standard library for loading and fine-tuning open-weight models.
    *   **[PEFT](https://huggingface.co/docs/peft)** — HuggingFace's Parameter-Efficient Fine-Tuning library (LoRA, QLoRA, prefix tuning).
    *   **[TRL](https://huggingface.co/docs/trl)** — Transformer Reinforcement Learning; includes SFT, DPO, and PPO trainers built on top of PEFT.
*   **Evaluation:**
    *   **[RAGAS](https://docs.ragas.io)** — automated evaluation framework for RAG pipelines (faithfulness, answer relevancy, context precision).
    *   **[LangSmith](https://smith.langchain.com)** — tracing, debugging, and evaluation platform for LangChain applications.
    *   **[Weights & Biases](https://wandb.ai)** — experiment tracking and model evaluation; widely used for fine-tuning runs.
*   **Videos:**
    *   [LangChain for LLM Application Development — DeepLearning.AI](https://www.youtube.com/watch?v=_v_fgW2SkkQ)
    *   [LlamaIndex — Full Crash Course](https://www.youtube.com/watch?v=cNMYeW2mpBs)
    *   [Hugging Face Transformers Course](https://www.youtube.com/watch?v=9HFw1VxiK1g)
    *   [Ollama Tutorial — Run LLMs Locally](https://www.youtube.com/watch?v=1IQGpRZfxjk)

### Practical Code Examples
*   **[RAG Pipeline](./resources/Python%20Code/RAG%20Pipeline/)** — Retrieval-Augmented Generation with local BERT, OpenAI, Anthropic, and Ollama backends.
*   **[Agent Tool Use](./resources/Python%20Code/Agent%20Tool%20Use/)** — Function calling / tool use with OpenAI and Anthropic.
*   **[Structured Output](./resources/Python%20Code/Structured%20Output/)** — JSON Schema extraction with OpenAI (strict mode), Anthropic (tool use pattern), and Ollama.
*   **[LoRA Fine-Tuning](./resources/Python%20Code/LoRA%20Fine-Tuning/)** — QLoRA fine-tuning with HuggingFace PEFT on a local model.

### Manim Animations

Programmatic animations of concepts that benefit most from visualisation.
Built with [Manim Community Edition](https://www.manim.community/) — the same library used by 3Blue1Brown.

> **Setup:** requires `pip install manim` plus FFmpeg and a LaTeX distribution.
> See [Manim Animations/readme.md](./resources/Manim%20Animations/readme.md) for full install instructions.

| Animation | Concept | Render command |
|---|---|---|
| [lora_rank_decomposition.py](./resources/Manim%20Animations/lora_rank_decomposition.py) | How A×B approximates a weight update | `manim -ql lora_rank_decomposition.py LoRARankDecomposition` |
| [rag_pipeline.py](./resources/Manim%20Animations/rag_pipeline.py) | Full index → retrieve → generate flow | `manim -ql rag_pipeline.py RAGPipeline` |
| [tokenization_bpe.py](./resources/Manim%20Animations/tokenization_bpe.py) | BPE merges turning characters into tokens | `manim -ql tokenization_bpe.py TokenizationBPE` |
| [moe_expert_routing.py](./resources/Manim%20Animations/moe_expert_routing.py) | Sparse top-k routing in Mixtral-style MoE | `manim -ql moe_expert_routing.py MoEExpertRouting` |

---

## Doctoral Level & Research

### Surveys and Overviews
*   **[A Survey of Large Language Models (2023)](./resources/KeyPapers/Survey_of_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2303.18223))
*   **[A Comprehensive Overview of Large Language Models (2023)](./resources/KeyPapers/Comprehensive_Overview_of_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2307.06435))
*   **[Evaluating Large Language Models: A Comprehensive Survey (2023)](./resources/KeyPapers/Evaluating_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2310.19736))
*   **[Efficient Large Language Models: A Survey (2023)](./resources/KeyPapers/Efficient_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2312.03863))
*   **[Annotated History of Modern AI and Deep Learning (2022) by Schmidhuber](./resources/KeyPapers/Annotated_History_of_Modern_AI_2022.pdf)** ([External](https://arxiv.org/abs/2212.11279))

### Specialized Topics
*   **[GPT-4 Technical Report (2023) by OpenAI](./resources/KeyPapers/GPT-4_Technical_Report_2023.pdf)** ([External](https://arxiv.org/abs/2303.08774)) — Architecture, capabilities, and alignment of GPT-4.
*   **[LLaMA 2 (2023) by Touvron et al.](./resources/KeyPapers/LLaMA_2_2023.pdf)** ([External](https://arxiv.org/abs/2307.09288)) — Open-weight models 7B–70B with safety fine-tuning; foundation for most open-source LLM work.
*   **[Mixtral of Experts (2024) by Mistral AI](./resources/KeyPapers/Mixtral_of_Experts_2024.pdf)** ([External](https://arxiv.org/abs/2401.04088)) — Sparse mixture-of-experts; strong performance with less active compute.
*   **[Mamba: Linear-Time Sequence Modeling (2023) by Gu & Dao](./resources/KeyPapers/Mamba_Linear-Time_Sequence_Modeling_2023.pdf)** ([External](https://arxiv.org/abs/2312.00752)) — State-space model alternative to transformers; linear scaling with sequence length.
*   **[Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model)**
*   **[Alignment faking in large language models](https://www.anthropic.com/research/alignment-faking)**

---

## Key Contributors & Resources

### Key Personalities & Labs
*   **Andrej Karpathy**: Former Director of AI at Tesla, known for his excellent explanations of deep learning concepts.
*   **Andrew Ng**: Co-founder of Google Brain and Coursera, a leading educator in AI and machine learning.
*   **Anthropic**: An AI safety and research company, focused on building reliable, interpretable, and steerable AI systems.
*   **DeepMind**: A subsidiary of Google, known for cutting-edge research in deep learning and reinforcement learning.
*   **Hugging Face**: A company and community building tools for the future of AI, most notably the Transformers library.

### YouTube Channels
*   **[Andrej Karpathy](https://www.youtube.com/@karpathy)**: In-depth lectures on building neural networks from scratch.
*   **[3Blue1Brown](https://www.youtube.com/@3blue1brown)**: Visual and intuitive explanations of complex math topics relevant to AI.
*   **[DeepLearning.AI](https://www.youtube.com/@DeepLearningAI)**: Courses and specializations from Andrew Ng and other experts.
*   **[FreeCodeCamp](https://www.youtube.com/@freecodecamp)**: Comprehensive tutorials on Python, PyTorch, and building LLMs.
*   **[Stanford Online](https://www.youtube.com/@stanfordonline)**: University-level courses on AI, deep learning, and NLP.

---

## How to Contribute

If you know of additional high-quality, freely available resources, please open a pull request or an issue.

## Glossary

**Artificial Intelligence (AI):** The field of creating systems that can perform tasks typically requiring human intelligence, such as reasoning, learning, and problem-solving.
**Machine Learning (ML):** A subset of AI focused on algorithms that learn patterns from data to make predictions or decisions without being explicitly programmed.
**Deep Learning:** A branch of ML using neural networks with many layers to model complex patterns in data.
**Neural Network:** A computational model inspired by the human brain, consisting of interconnected nodes (neurons) that process information in layers.
**Transformer:** A neural network architecture based on self-attention mechanisms, enabling efficient modeling of sequential data. Introduced in "Attention Is All You Need" (2017).
**Attention Mechanism:** A technique allowing models to focus on relevant parts of input sequences when making predictions, crucial for transformers.
**Word Embedding:** A representation of words as dense vectors in continuous space, capturing semantic relationships (e.g., Word2Vec, GloVe).
**Pre-training:** Training a model on a large, generic dataset to learn general features before fine-tuning on a specific task.
**Fine-tuning:** Adapting a pre-trained model to a specific task or dataset by continuing training on new data.
**Large Language Model (LLM):** A neural network trained on vast text corpora to generate and understand human language (e.g., GPT, BERT, T5).
**BERT:** Bidirectional Encoder Representations from Transformers; a transformer-based model pre-trained for language understanding tasks.
**GPT:** Generative Pre-trained Transformer; a family of transformer-based models for text generation and understanding (e.g., GPT-2, GPT-3).
**Few-shot Learning:** The ability of a model to generalize to new tasks given only a few examples.
**RLHF (Reinforcement Learning from Human Feedback):** A training approach where models are optimized using feedback from human evaluators, often for alignment and safety.
**LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method for large models, using low-rank updates to reduce resource requirements.
**FlashAttention:** An efficient implementation of the attention mechanism that reduces memory usage and increases speed.
**Chain-of-Thought Prompting:** A prompting technique that encourages models to reason step-by-step, improving performance on complex tasks.
**Capstone Project:** A comprehensive, practical project that integrates and applies learned skills, often as a final challenge in a curriculum.
**Dataset:** A collection of data used for training or evaluating machine learning models.
**Tokenization:** The process of breaking text into smaller units (tokens), such as words or subwords, for model input.
**Supervised Learning:** ML where models are trained on labeled data (input-output pairs).
**Unsupervised Learning:** ML where models find patterns in unlabeled data.
**Transfer Learning:** Leveraging knowledge from one task or dataset to improve performance on another, often via pre-training and fine-tuning.
**Overfitting:** When a model learns noise or details specific to the training data, reducing its ability to generalize.
**Underfitting:** When a model is too simple to capture underlying patterns in the data, resulting in poor performance.
**RAG (Retrieval-Augmented Generation):** A pattern where a retrieval system fetches relevant documents at query time and passes them as context to an LLM, grounding responses in external knowledge without retraining.
**Vector Database:** A database optimised for storing and searching high-dimensional embedding vectors by similarity (e.g., Chroma, Qdrant, Pinecone, pgvector).
**Embedding:** A dense numerical vector representing the meaning of a piece of text, image, or other data; similar items have vectors that are close together in the embedding space.
**AI Agent:** An LLM system that can take actions — calling tools, APIs, or other models — and iterate through a reasoning loop to complete multi-step goals autonomously.
**Tool Use / Function Calling:** A capability where a model outputs structured calls to external functions (e.g., search, calculator, database query) and incorporates their results into its reasoning.
**DPO (Direct Preference Optimization):** A fine-tuning method that optimises model outputs to match human preferences directly, without training a separate reward model as in RLHF.
**QLoRA:** Combines 4-bit quantization of the base model with LoRA adapters, dramatically reducing GPU memory requirements for fine-tuning large models.
**Mixture of Experts (MoE):** A model architecture where different subsets of parameters ("experts") are activated for different inputs, allowing a very large total parameter count while keeping per-token compute constant.
**Quantization:** Reducing the numerical precision of model weights (e.g., from 32-bit float to 4-bit integer) to shrink memory footprint and speed up inference, with a small quality tradeoff.
**Prompt Engineering:** The practice of crafting input prompts to reliably elicit desired behaviours from an LLM, without changing the model's weights.
**Context Window:** The maximum number of tokens an LLM can process in a single call, encompassing both the input prompt and the generated output.
**Structured Output:** Constraining an LLM's response to a specific format (typically JSON) that can be parsed programmatically, using JSON Schema, tool use, or format parameters.
**Multimodal Model:** An AI model that accepts more than one type of input — typically text combined with images, audio, or video — and reasons over them jointly to produce a response.
**Vision Encoder:** A neural network (often a Vision Transformer or CNN) that converts an image into a sequence of embedding vectors that a language model can process alongside text tokens.
**CLIP:** Contrastive Language-Image Pretraining; a model trained to align image and text embeddings so that an image and its description are close together in the same vector space.
**Token:** The basic unit of text that a language model processes; typically a word fragment (e.g. "transform" + "er"). Most models use Byte-Pair Encoding (BPE) to build their vocabulary of 30k–100k+ tokens.

## License

This repository contains links to publicly available educational resources. The individual resources maintain their own licenses. Please check individual resources for their specific licensing terms.