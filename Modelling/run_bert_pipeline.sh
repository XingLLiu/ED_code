#!/bin/sh
# ----------- MASTER SHELL FOR ED BERT TRAINING/TESING ----------- #
# This script calls and runs the Clinical BERT prediction pipeline.
# To run:
# in Terminal:
#   sh run_bert_pipeline.sh
# ---------------------------------------------------------------- #

echo "$(tput setaf 1)Running BERT pipeline...$(tput sgr 0)"

# Fine-tune and save models
for j in 2 3 4 5 6 7 8 9 10
    do
        echo "$(tput setaf 1)Start fine-tuning Clinical BERT for $j ...$(tput sgr 0)"
        python bert_pipeline.py \
        --path=$/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv \
        --task_name=epic_task \
        --start_time=$j
        echo "$(tput setaf 1) $j completed.\n$(tput sgr 0)"
    done


