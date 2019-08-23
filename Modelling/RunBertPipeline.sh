# Run clinical BERT
# ----------- MASTER SHELL FOR ED BERT TRAINING/TESING ----------- #

CODE_DIR=$(pwd)

echo "$(tput setaf 1)Contents of output directory:$(tput sgr 0)"

# Fine-tune and save model
for j in 2 3 4 5 6 7 8 9 10
    do
        echo "$(tput setaf 1)Start fine-tuning Clinical BERT for $j ...$(tput sgr 0)"
        python bert_pipeline.py \
        --path=$/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv \
        --task_name=epic_task \
        --start_time=$j
        echo "$(tput setaf 1) $j completed.\n$(tput sgr 0)"

        # Zip file for the next iteration

    done



cd ../../results/bert/dynamic/201808/Saved_Checkpoints/epic_task/
tar -czvf epic_task.tar.gz bert_config.json pytorch_model.bin

