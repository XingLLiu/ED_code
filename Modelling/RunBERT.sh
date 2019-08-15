# Run clinical BERT
# ----------- MASTER SHELL FOR ED BERT TRAINING/TESING ----------- #

OUTPUT_DIR=${HOME}/Documents/ED/data/EPIC_DATA/EPIC_for_Bert/Saved_Checkpoints/epic_task
CODE_DIR=$(pwd)

echo "$(tput setaf 1)Contents of output directory:$(tput sgr 0)"
ls $OUTPUT_DIR

# Fine-tune and save model
echo "$(tput setaf 1)Start fine-tuning Clinical BERT ...$(tput sgr 0)"
python ModelsBERT.py \
--clean_notes=True \
--mode train \
--path=$/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv \
--task_name=epic_task
echo "$(tput setaf 1)Complete.\n$(tput sgr 0)"

# Compress the fine-tuned model
echo "$(tput setaf 1)Zipping fine-tuned model ...$(tput sgr 0)"
cd $OUTPUT_DIR
tar -czvf epic_task.tar.gz bert_config.json pytorch_model.bin
cd $CODE_DIR
echo "$(tput setaf 1)Complete.\n$(tput sgr 0)"

# Test the model
echo "$(tput setaf 1)Start testing the model ...$(tput sgr 0)"
python ModelsBERT.py \
--clean_notes=False \
--mode test \
--path=$/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv \
--task_name=epic_task
echo "$(tput setaf 1)Complete.\n$(tput sgr 0)"


