@echo off
REM Download key AI/ML papers as PDFs.
REM Run from the KeyPapers directory:  download_pdfs.bat
REM Requires: curl (built into Windows 10 1803+)

cd /d "%~dp0"

echo === Foundational Papers ===
call :download "https://arxiv.org/pdf/1301.3781"  "Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf"
call :download "https://arxiv.org/pdf/1706.03762" "Attention_Is_All_You_Need_2017.pdf"
call :download "https://arxiv.org/pdf/1810.04805" "BERT_2018.pdf"
call :download "https://arxiv.org/pdf/1901.02860" "Transformer-XL_2019.pdf"
call :download "https://arxiv.org/pdf/2003.10555" "ELECTRA_2020.pdf"
call :download "https://arxiv.org/pdf/2005.14165" "GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf"
call :download "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" "Improving_Language_Understanding_by_Generative_Pre-Training_2018.pdf"
call :download "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" "Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf"
call :download "https://arxiv.org/pdf/1910.10683" "Exploring_the_Limits_of_Transfer_Learning_with_a_Unified_Text-to-Text_Transformer_2020.pdf"

echo.
echo === Key Methods ===
call :download "https://arxiv.org/pdf/2106.09685" "LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf"
call :download "https://arxiv.org/pdf/2203.02155" "Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf"
call :download "https://arxiv.org/pdf/2205.14135" "FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf"
call :download "https://arxiv.org/pdf/2201.11903" "Chain-of-Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models_2022.pdf"

echo.
echo === Surveys and Overviews ===
call :download "https://arxiv.org/pdf/2212.11279" "Annotated_History_of_Modern_AI_2022.pdf"
call :download "https://arxiv.org/pdf/2307.06435" "Comprehensive_Overview_of_LLMs_2023.pdf"
call :download "https://arxiv.org/pdf/2310.19736" "Evaluating_LLMs_2023.pdf"
call :download "https://arxiv.org/pdf/2312.03863" "Efficient_LLMs_2023.pdf"
call :download "https://arxiv.org/pdf/2303.18223" "Survey_of_LLMs_2023.pdf"
call :download "https://arxiv.org/pdf/2212.09561" "Formal_Aspects_of_Language_Modeling_2023.pdf"
call :download "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf" "A_Neural_Probabilistic_Language_Model_2003.pdf"

echo.
echo === New Papers (2023-2024) ===
call :download "https://arxiv.org/pdf/2303.08774" "GPT-4_Technical_Report_2023.pdf"
call :download "https://arxiv.org/pdf/2307.09288" "LLaMA_2_2023.pdf"
call :download "https://arxiv.org/pdf/2401.04088" "Mixtral_of_Experts_2024.pdf"
call :download "https://arxiv.org/pdf/2005.11401" "RAG_Retrieval-Augmented_Generation_2020.pdf"
call :download "https://arxiv.org/pdf/2212.08073" "Constitutional_AI_Harmlessness_from_AI_Feedback_2022.pdf"
call :download "https://arxiv.org/pdf/2305.18290" "DPO_Direct_Preference_Optimization_2023.pdf"
call :download "https://arxiv.org/pdf/2305.14314" "QLoRA_Efficient_Finetuning_of_Quantized_LLMs_2023.pdf"
call :download "https://arxiv.org/pdf/2312.00752" "Mamba_Linear-Time_Sequence_Modeling_2023.pdf"

echo.
echo All downloads complete.
goto :eof

:download
if exist "%~2" (
    echo   [skip] %~2 already exists
) else (
    echo   [download] %~2
    curl -L --retry 3 -o "%~2" "%~1"
)
goto :eof
