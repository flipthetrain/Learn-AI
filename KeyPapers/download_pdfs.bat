@echo off
REM Batch script to download all external PDF links from Recommended Reading Order
REM Requires PowerShell (included in Windows 10+)

setlocal enabledelayedexpansion

REM Create KeyPapers directory if it doesn't exist
if not exist "KeyPapers" mkdir "KeyPapers"

REM Download each PDF
powershell -Command "Invoke-WebRequest -Uri 'https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf' -OutFile 'KeyPapers\A_Neural_Probabilistic_Language_Model_2003.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1301.3781.pdf' -OutFile 'KeyPapers\Efficient_Estimation_of_Word_Representations_in_Vector_Space_2013.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1706.03762.pdf' -OutFile 'KeyPapers\Attention_Is_All_You_Need_2017.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1901.02860.pdf' -OutFile 'KeyPapers\Transformer-XL_2019.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1810.04805.pdf' -OutFile 'KeyPapers\BERT_2018.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2005.14165.pdf' -OutFile 'KeyPapers\GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2203.02155.pdf' -OutFile 'KeyPapers\Training_language_models_to_follow_instructions_with_human_feedback_2022.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2106.09685.pdf' -OutFile 'KeyPapers\LoRA_Low-Rank_Adaptation_of_Large_Language_Models_2021.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2205.14135.pdf' -OutFile 'KeyPapers\FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_2022.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2003.10555.pdf' -OutFile 'KeyPapers\ELECTRA_2020.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2201.11903.pdf' -OutFile 'KeyPapers\Chain-of-Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models_2022.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2212.11279.pdf' -OutFile 'KeyPapers\Annotated_History_of_Modern_AI_2022.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2303.18223.pdf' -OutFile 'KeyPapers\Survey_of_LLMs_2023.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2307.06435.pdf' -OutFile 'KeyPapers\Comprehensive_Overview_of_LLMs_2023.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2310.19736.pdf' -OutFile 'KeyPapers\Evaluating_LLMs_2023.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2312.03863.pdf' -OutFile 'KeyPapers\Efficient_LLMs_2023.pdf'"
powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2311.04329.pdf' -OutFile 'KeyPapers\Formal_Aspects_of_Language_Modeling_2023.pdf'"

endlocal

echo All downloads attempted. Check the PDF folder for results.
pause
