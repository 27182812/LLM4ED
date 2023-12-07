#!/bin/bash


# set python path according to your actual environment
pythonpath='python'

# few-shot ICL
${pythonpath} prompt_fewshot.py --model "gpt-3.5-turbo" --save_path "results/ED_chatgpt_shot5.txt" --few_shot 5 --apikey "your own openai.api_key"

# Semantically Similar ICL
# ${pythonpath} prompt_sensim.py --model "gpt-3.5-turbo" --save_path "results/ED_chatgpt_shot5_sensim.txt" --few_shot 5 --apikey "your own openai.api_key"

# Two-stage Interactive Generation
# ${pythonpath} prompt_twostages.py --model "gpt-3.5-turbo" --save_path "results/ED_chatgpt_stage1.txt" --satge 1 --apikey "your own openai.api_key"
# ${pythonpath} prompt_twostages.py --model "gpt-3.5-turbo" --save_path "results/ED_chatgpt_stage2.txt" --stage 2 --apikey "your own openai.api_key"
  
# Knowledge Base
# ${pythonpath} prompt_knowledge.py --model "gpt-3.5-turbo" --save_path "results/ED_chatgpt_knowledge.txt" --apikey "your own openai.api_key"