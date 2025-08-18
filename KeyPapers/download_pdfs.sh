#!/bin/bash
# Linux shell script to download all external PDF links from Recommended Reading Order
# Requires wget

mkdir -p KeyPapers

wget -O KeyPapers/A_Neural_Probabilistic_Language_Model_2003.pdf "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"
wget -O KeyPapers/Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf "https://arxiv.org/pdf/1301.3781.pdf"
wget -O KeyPapers/Attention_Is_All_You_Need_2017.pdf "https://arxiv.org/pdf/1706.03762.pdf"
wget -O KeyPapers/Transformer-XL_2019.pdf "https://arxiv.org/pdf/1901.02860.pdf"
wget -O KeyPapers/BERT_2018.pdf "https://arxiv.org/pdf/1810.04805.pdf"
wget -O KeyPapers/GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf "https://arxiv.org/pdf/2005.14165.pdf"
wget -O KeyPapers/Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf "https://arxiv.org/pdf/2203.02155.pdf"
wget -O KeyPapers/LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf "https://arxiv.org/pdf/2106.09685.pdf"
wget -O KeyPapers/FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf "https://arxiv.org/pdf/2205.14135.pdf"
wget -O KeyPapers/ELECTRA_2020.pdf "https://arxiv.org/pdf/2003.10555.pdf"
wget -O KeyPapers/Chain-of-Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models_2022.pdf "https://arxiv.org/pdf/2201.11903.pdf"
wget -O KeyPapers/Annotated_History_of_Modern_AI_2022.pdf "https://arxiv.org/pdf/2212.11279.pdf"
wget -O KeyPapers/Survey_of_LLMs_2023.pdf "https://arxiv.org/pdf/2303.18223.pdf"
wget -O KeyPapers/Comprehensive_Overview_of_LLMs_2023.pdf "https://arxiv.org/pdf/2307.06435.pdf"
wget -O KeyPapers/Evaluating_LLMs_2023.pdf "https://arxiv.org/pdf/2310.19736.pdf"
wget -O KeyPapers/Efficient_LLMs_2023.pdf "https://arxiv.org/pdf/2312.03863.pdf"
wget -O KeyPapers/Formal_Aspects_of_Language_Modeling_2023.pdf "https://arxiv.org/pdf/2311.04329.pdf"

echo "All downloads attempted. Check the PDF folder for results."
