# AI Curriculum

This document provides a curated list of resources in the fields of Artificial Intelligence (AI) and Large Language Models (LLMs), organized into three main sections: AI Papers, AI Videos, and Essential Mathematics.

# Part 1: AI Papers

## Reading Guide

Before diving into the papers, check the "Recommended Reading Order" section at the end of the Papers section for a structured learning path.

## Foundational Papers

* **[A Neural Probabilistic Language Model (2003) by Bengio et al.](./KeyPapers/A_Neural_Probabilistic_Language_Model_2003.pdf)** ([External](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)) — Introduced distributed word representations and early neural language modeling.
* **[Efficient Estimation of Word Representations in Vector Space (2013) by Mikolov et al.](./KeyPapers/Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf)** ([External](https://arxiv.org/abs/1301.3781)) — Word2Vec; efficient word embeddings.
* **[Attention Is All You Need (2017) by Vaswani et al.](./KeyPapers/Attention_Is_All_You_Need_2017.pdf)** ([External](https://arxiv.org/abs/1706.03762)) — Introduced the Transformer architecture.
* **[BERT: Pre-training of Deep Bidirectional Transformers (2018) by Devlin et al.](./KeyPapers/BERT_2018.pdf)** ([External](https://arxiv.org/abs/1810.04805)) — Bidirectional pre-training for representation learning.
* **[Improving Language Understanding by Generative Pre-Training (2018) by Radford et al.](./KeyPapers/Improving_Language_Understanding_by_Generative_Pre-Training_2018.pdf)** ([External](https://openai.com/research/language-unsupervised)) — Early GPT demonstration of generative pre-training.
* **[Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](./KeyPapers/Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf)** ([External](https://openai.com/research/better-language-models)) — Scaling LMs for generalization.
* **[Language Models are Few-Shot Learners (GPT-3, 2020) by Brown et al.](./KeyPapers/GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf)** ([External](https://arxiv.org/abs/2005.14165)) — Few-shot capabilities emerge at scale.
* **[Transformer-XL (2019) by Dai et al.](./KeyPapers/Transformer-XL_2019.pdf)** ([External](https://arxiv.org/abs/1901.02860)) — Long-context modeling with recurrence and relative positions.
* **[T5: Text-to-Text Transfer Transformer (2020) by Raffel et al.](./KeyPapers/Exploring_the_Limits_of_Transfer_Learning_with_a_Unified_Text-to-Text_Transformer_2020.pdf)** ([External](https://arxiv.org/abs/1910.10683)) — Unified text-to-text framework.
* **[ELECTRA (2020) by Clark et al.](./KeyPapers/ELECTRA_2020.pdf)** ([External](https://arxiv.org/abs/2003.10555)) — Sample-efficient pre-training via replaced-token detection.

## Key Methods & Optimizations

* **[Training language models to follow instructions with human feedback (2022)](./KeyPapers/Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf)** ([External](https://arxiv.org/abs/2203.02155)) — RLHF for alignment and helpfulness.
* **[LoRA: Low-Rank Adaptation of Large Language Models (2021)](./KeyPapers/LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf)** ([External](https://arxiv.org/abs/2106.09685)) — Parameter-efficient fine-tuning.
* **[FlashAttention (2022) by Dao et al.](./KeyPapers/FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf)** ([External](https://arxiv.org/abs/2205.14135)) — Fast, memory-efficient attention implementation.
* **[Chain-of-Thought Prompting (2022) by Wei et al.](./KeyPapers/Chain-of-Thought_Prompting_Elicits_Reasoningin_Large_Language_Models_2022.pdf)** ([External](https://arxiv.org/abs/2201.11903)) — Improves reasoning via stepwise prompting.

## Anthropic Papers on Language Models

* **[Constitutional AI: Harmlessness from AI Feedback](./AnthropicPapers/Constitutional_AI_Harmlessness_from_AI_Feedback_2022.pdf)** ([External](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback))
* **[Collective Constitutional AI: Aligning a Language Model with Public Input](./AnthropicPapers/Collective_Constitutional_AI_2023.pdf)** ([External](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input))
* **[Discovering Language Model Behaviors with Model-Written Evaluations](./AnthropicPapers/Discovering_LM_Behaviors_with_Model_Written_Evaluations_2022.pdf)** ([External](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations))
* **[Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers)** — Defensive classifiers for jailbreaks.
* **[Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model)**
* **[Reasoning models don&#39;t always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think)**
* **[Alignment faking in large language models](https://www.anthropic.com/research/alignment-faking)**
* **[Claude 3 Model Family (2024)](./AnthropicPapers/Claude3_Model_Family_2024.pdf)** ([External](https://www.anthropic.com/news/claude-3-family))

## Surveys and Overviews

* **[A Survey of Large Language Models (2023)](./LanguageModelPapers/Survey_of_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2303.18223))
* **[A Comprehensive Overview of Large Language Models (2023)](./LanguageModelPapers/Comprehensive_Overview_of_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2307.06435))
* **[Evaluating Large Language Models: A Comprehensive Survey (2023)](./LanguageModelPapers/Evaluating_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2310.19736))
* **[Efficient Large Language Models: A Survey (2023)](./LanguageModelPapers/Efficient_LLMs_2023.pdf)** ([External](https://arxiv.org/abs/2312.03863))
* **[Formal Aspects of Language Modeling (2023)](./LanguageModelPapers/Formal_Aspects_of_Language_Modeling_2023.pdf)** ([External](https://arxiv.org/abs/2311.04329))
* **[Annotated History of Modern AI and Deep Learning (2022) by Schmidhuber](./KeyPapers/Annotated_History_of_Modern_AI_2022.pdf)** ([External](https://arxiv.org/abs/2212.11279))

## Recommended Reading Order

1. Foundational concepts:
   - [A Neural Probabilistic Language Model (2003) by Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) ([local](./PDF/A_Neural_Probabilistic_Language_Model_2003.pdf))
   - [Efficient Estimation of Word Representations in Vector Space (2013) by Mikolov et al.](https://arxiv.org/abs/1301.3781) ([local](./PDF/Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf))
2. Transformer and variants:
   - [Attention Is All You Need (2017) by Vaswani et al.](https://arxiv.org/abs/1706.03762) ([local](./PDF/Attention_Is_All_You_Need_2017.pdf))
   - [Transformer-XL (2019) by Dai et al.](https://arxiv.org/abs/1901.02860) ([local](./PDF/Transformer-XL_2019.pdf))
   - [BERT: Pre-training of Deep Bidirectional Transformers (2018) by Devlin et al.](https://arxiv.org/abs/1810.04805) ([local](./PDF/BERT_2018.pdf))
3. Scaling and GPT series:
   - [Improving Language Understanding by Generative Pre-Training (2018) by Radford et al.](https://openai.com/research/language-unsupervised)
   - [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://openai.com/research/better-language-models)
   - [Language Models are Few-Shot Learners (GPT-3, 2020) by Brown et al.](https://arxiv.org/abs/2005.14165) ([local](./PDF/GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf))
4. Instruction tuning & alignment:
   - [Training language models to follow instructions with human feedback (2022)](https://arxiv.org/abs/2203.02155) ([local](./PDF/Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf))
5. Efficiency & fine-tuning:
   - [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685) ([local](./PDF/LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf))
   - [FlashAttention (2022) by Dao et al.](https://arxiv.org/abs/2205.14135) ([local](./PDF/FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf))
   - [ELECTRA (2020) by Clark et al.](https://arxiv.org/abs/2003.10555) ([local](./PDF/ELECTRA_2020.pdf))
6. Reasoning techniques:
   - [Chain-of-Thought Prompting (2022) by Wei et al.](https://arxiv.org/abs/2201.11903) ([local](./PDF/Chain-of-Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models_2022.pdf))
7. Surveys & history:
   - [Annotated History of Modern AI and Deep Learning (2022) by Schmidhuber](https://arxiv.org/abs/2212.11279) ([local](./PDF/Annotated_History_of_Modern_AI_2022.pdf))
   - [A Survey of Large Language Models (2023)](https://arxiv.org/abs/2303.18223) ([local](./PDF/Survey_of_LLMs_2023.pdf))
   - [A Comprehensive Overview of Large Language Models (2023)](https://arxiv.org/abs/2307.06435) ([local](./PDF/Comprehensive_Overview_of_LLMs_2023.pdf))
   - [Evaluating Large Language Models: A Comprehensive Survey (2023)](https://arxiv.org/abs/2310.19736) ([local](./PDF/Evaluating_LLMs_2023.pdf))
   - [Efficient Large Language Models: A Survey (2023)](https://arxiv.org/abs/2312.03863) ([local](./PDF/Efficient_LLMs_2023.pdf))
   - [Formal Aspects of Language Modeling (2023)](https://arxiv.org/abs/2311.04329) ([local](./PDF/Formal_Aspects_of_Language_Modeling_2023.pdf))
8. Safety & interpretability:
   - [Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
   - [Collective Constitutional AI: Aligning a Language Model with Public Input](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)
   - [Discovering Language Model Behaviors with Model-Written Evaluations](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations)
   - [Claude 3 Model Family (2024)](https://www.anthropic.com/news/claude-3-family)

---

# Part 2: AI Videos

### Deep Learning Foundations

* **[Neural Networks from Scratch — Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)**
* **[Deep Learning Fundamentals — DeepLearning.AI](https://www.youtube.com/watch?v=pyqt7s2bpqM)**
* **[Attention and Transformer Networks — Stanford CS224N](https://www.youtube.com/watch?v=ptuGllU5SQQ)**

### Language Model Development

* **[Stanford CS229: Building Large Language Models — Stanford Online](https://www.youtube.com/watch?v=9vM4p9NN0Ts)**
* **[Developing Large Language Models in Python — NeuralNine](https://www.youtube.com/watch?v=s5nq-a1wpPY)**
* **[Create a Large Language Model from Scratch — FreeCodeCamp](https://www.youtube.com/watch?v=UU1WVnMk4E8)**
* **[How Transformers Work — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)**
* **[The Transformer Architecture — Stanford CS224N](https://www.youtube.com/watch?v=S27pHKBEp30)**

### Fine-tuning and Advanced Topics

* **[Fine-tuning LLMs w/ Example Code — Shawhin Talebi](https://www.youtube.com/watch?v=eC6Hd1hFvos)**
* **[RLHF: Training Language Models with Human Feedback — Hugging Face](https://www.youtube.com/watch?v=2MBJOuVq380)**
* **[Advanced LLM Training Techniques — DeepLearning.AI](https://www.youtube.com/watch?v=yRB_b-S6ZtI)**
* **[Efficient Fine-tuning of Language Models — Microsoft Research](https://www.youtube.com/watch?v=9ZqE0HhXaVU)**

### LangChain Development

* **[LangChain for LLM Application Development — DeepLearning.AI](https://www.youtube.com/watch?v=_v_fgW2SkkQ)**
* **[LangChain Crash Course — Patrick Loeber](https://www.youtube.com/watch?v=nE2skSRWTTs)**
* **[Build LLM Apps with LangChain — FreeCodeCamp](https://www.youtube.com/watch?v=lG7Uxts9SXs)**
* **[LangChain Series: Building Production Apps — James Briggs](https://www.youtube.com/watch?v=MlK6SIjcjE8)**
* **[LangChain Agents Tutorial — Tech With Tim](https://www.youtube.com/watch?v=jz-lMRJXbNw)**

## Recommended AI Video Viewing Order

1. **Programming Prerequisites**:
   - [Python Full Course for Beginners [2025] — FreeCodeCamp](https://www.youtube.com/watch?v=K5KVEU3aaeQ)
   - [Python for Beginners — FreeCodeCamp](https://www.youtube.com/watch?v=eWRfhZUzrAc)
   - [Python NumPy Tutorial — freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI)
   - [PyTorch for Deep Learning — freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA)
2. **Deep Learning Prerequisites**:
   - [Neural Networks from Scratch — Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
   - [Deep Learning Fundamentals — DeepLearning.AI](https://www.youtube.com/watch?v=pyqt7s2bpqM)
   - [Attention and Transformer Networks — Stanford CS224N](https://www.youtube.com/watch?v=ptuGllU5SQQ)
3. **Core LLM Concepts**:
   - [How Transformers Work — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
   - [The Transformer Architecture — Stanford CS224N](https://www.youtube.com/watch?v=S27pHKBEp30)
   - [Stanford CS229: Building Large Language Models — Stanford Online](https://www.youtube.com/watch?v=9vM4p9NN0Ts)
4. **Practical Implementation**:
   - [Developing Large Language Models in Python — NeuralNine](https://www.youtube.com/watch?v=s5nq-a1wpPY)
   - [Create a Large Language Model from Scratch — FreeCodeCamp](https://www.youtube.com/watch?v=UU1WVnMk4E8)
5. **Advanced Topics**:
   - [Fine-tuning LLMs w/ Example Code — Shawhin Talebi](https://www.youtube.com/watch?v=eC6Hd1hFvos)
   - [RLHF: Training Language Models with Human Feedback — Hugging Face](https://www.youtube.com/watch?v=2MBJOuVq380)
   - [Advanced LLM Training Techniques — DeepLearning.AI](https://www.youtube.com/watch?v=yRB_b-S6ZtI)
   - [Efficient Fine-tuning of Language Models — Microsoft Research](https://www.youtube.com/watch?v=9ZqE0HhXaVU)
6. **LangChain Development**:
   - [LangChain Crash Course — Patrick Loeber](https://www.youtube.com/watch?v=nE2skSRWTTs)
   - [LangChain for LLM Application Development — DeepLearning.AI](https://www.youtube.com/watch?v=_v_fgW2SkkQ)
   - [Build LLM Apps with LangChain — FreeCodeCamp](https://www.youtube.com/watch?v=lG7Uxts9SXs)
   - [LangChain Series: Building Production Apps — James Briggs](https://www.youtube.com/watch?v=MlK6SIjcjE8)
   - [LangChain Agents Tutorial — Tech With Tim](https://www.youtube.com/watch?v=jz-lMRJXbNw)

---

# Part 3: AI Ethics

## Recommended AI Ethics Viewing Order

1. [AI Alignment and Safety — Anthropic](https://www.youtube.com/watch?v=FX0zYxm4yes)
2. [Ethics in AI Development — DeepMind](https://www.youtube.com/watch?v=z7tRnlqybnU)
3. [Responsible AI Development — Stanford HAI](https://www.youtube.com/watch?v=xI_xM9K8q4k)

---

# Part 4: Essential Mathematics

## Recommended Mathematics Viewing Order

1. **Core Mathematics**:
   - [Linear Algebra Fundamentals — 3Blue1Brown](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
   - [Calculus Essentials — 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
   - [Statistics and Probability — Khan Academy](https://www.youtube.com/watch?v=uhxtUt_-GyM&list=PL1328115D3D8A2566)
   - [Matrix Operations for Machine Learning — StatQuest](https://www.youtube.com/watch?v=NKGqFBLC4eI)
2. **Advanced Mathematics**:
   - [Optimization Mathematics — Stanford](https://www.youtube.com/watch?v=Q4L_zxDrcPU)
   - [Information Theory Basics — Stanford](https://www.youtube.com/watch?v=5c9zopx6Tqk)
3. **AI-Specific Mathematics**:
   - [Mathematics of Neural Networks — StatQuest](https://www.youtube.com/watch?v=CqOfi41LfDw)
   - [Backpropagation Mathematics — 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
   - [Mathematical Foundations of Machine Learning — MIT OpenCourseWare](https://www.youtube.com/watch?v=3eNzJGd9HgU)
   - [Probability in Machine Learning — StatQuest](https://www.youtube.com/watch?v=PrkiRVcrxOs)
   - [Information Theory in Machine Learning — Stanford](https://www.youtube.com/watch?v=9lyoqwT6DGY)
4. **Deep Learning Prerequisites**:
   - [Neural Networks from Scratch — Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
   - [Deep Learning Fundamentals — DeepLearning.AI](https://www.youtube.com/watch?v=pyqt7s2bpqM)
   - [Attention and Transformer Networks — Stanford CS224N](https://www.youtube.com/watch?v=ptuGllU5SQQ)
5. **Core LLM Concepts**:
   - [How Transformers Work — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
   - [The Transformer Architecture — Stanford CS224N](https://www.youtube.com/watch?v=S27pHKBEp30)
   - [Stanford CS229: Building Large Language Models — Stanford Online](https://www.youtube.com/watch?v=9vM4p9NN0Ts)
6. **Practical Implementation**:
   - [Developing Large Language Models in Python — NeuralNine](https://www.youtube.com/watch?v=s5nq-a1wpPY)
   - [Create a Large Language Model from Scratch — FreeCodeCamp](https://www.youtube.com/watch?v=UU1WVnMk4E8)
7. **Advanced Topics**:
   - [Fine-tuning LLMs w/ Example Code — Shawhin Talebi](https://www.youtube.com/watch?v=eC6Hd1hFvos)
   - [RLHF: Training Language Models with Human Feedback — Hugging Face](https://www.youtube.com/watch?v=2MBJOuVq380)
   - [Advanced LLM Training Techniques — DeepLearning.AI](https://www.youtube.com/watch?v=yRB_b-S6ZtI)
   - [Efficient Fine-tuning of Language Models — Microsoft Research](https://www.youtube.com/watch?v=9ZqE0HhXaVU)
8. **LangChain Development**:
   - [LangChain Crash Course — Patrick Loeber](https://www.youtube.com/watch?v=nE2skSRWTTs)
   - [LangChain for LLM Application Development — DeepLearning.AI](https://www.youtube.com/watch?v=_v_fgW2SkkQ)
   - [Build LLM Apps with LangChain — FreeCodeCamp](https://www.youtube.com/watch?v=lG7Uxts9SXs)
   - [LangChain Series: Building Production Apps — James Briggs](https://www.youtube.com/watch?v=MlK6SIjcjE8)
   - [LangChain Agents Tutorial — Tech With Tim](https://www.youtube.com/watch?v=jz-lMRJXbNw)

---

# Part 5: Essential Python

## Recommended Python Viewing Order

1. **Python Fundamentals**:
   - [Python Full Course for Beginners [2025] — FreeCodeCamp](https://www.youtube.com/watch?v=K5KVEU3aaeQ)
   - [Python for Beginners — FreeCodeCamp](https://www.youtube.com/watch?v=eWRfhZUzrAc)
   - [Python NumPy Tutorial — freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI)
   - [PyTorch for Deep Learning — freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA)

---

# Part 6: AI Tools & Frameworks

* **[Hugging Face Transformers Course](https://www.youtube.com/watch?v=9HFw1VxiK1g)**
* **[TensorFlow 2.0 Complete Course — freeCodeCamp](https://www.youtube.com/watch?v=tPYj3fFJGjk)**
* **[Scikit-learn Machine Learning Crash Course](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)**

---

# Part 7: Real-World AI Applications

* **[How AI is Used in Healthcare](https://www.youtube.com/watch?v=7D1CQ_LOizA)**
* **[Building a Chatbot with Transformers](https://www.youtube.com/watch?v=8Mpc9ukltVA)**

---

# Part 8: Capstone Projects & Challenges

* **[Kaggle Titanic Competition Walkthrough](https://www.youtube.com/watch?v=8A7TgG7E2XI)**
* **[How to Win a Kaggle Competition](https://www.youtube.com/watch?v=BH9FywlwKHo)**
* **[Awesome Open Datasets for Machine Learning](https://github.com/awesomedata/awesome-public-datasets)**

---

# Glossary

A section for key AI/ML terms. (To be expanded)

---

# Further Reading & News

* **[arXiv Sanity Preserver](http://www.arxiv-sanity.com/)**
* **[The Batch by Andrew Ng](https://www.deeplearning.ai/thebatch/)**
* **[AI for Everyone — Andrew Ng](https://www.youtube.com/watch?v=NKpuX_yzdYs)**

---

# Additional Papers

* **[DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)**
* **[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)**
* **[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)**
* **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)**
