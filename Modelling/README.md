# Clinical BERT Intall and Usage Guide

## Folder structure
There is no need to manually create any folders when running `ModelsBERT.py`, as long as the folder structure described in the root directory is followed. 

By default, all outputs from `ModelsBERT.py` will be stored in `/ED/data/EPIC_DATA/EPIC_for_Bert/` as follows:

```
|-- Processed_Notes   # Triage notes with tokens added
|-- Raw_Notes         # Cleaned triage notes
|-- Reports           # Evaluation plots and model predictions
    |__ Task_Name
|-- Saved_Checkpoints # Fine-tuned models
    |__ Task_Name
```

## Download Clinical BERT
Assuming the current directory is '/ED/ED_code/', the following code downloads the [Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT) ([E. Alsentzer *et al* 2019](https://arxiv.org/abs/1904.03323)) and unzip it into the folder `/ED/ClinicalBert/`:
```
mkdir ../ClinicalBert
wget -O ../ClinicalBert/pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
tar zxvf ../ClinicalBert/pretrained_bert_tf.tar.gz -C ../ClinicalBert/
```

## Fine-Tune Clinical BERT
BERT models require the input texts to be in a specific format. [Here](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04) is a clear and gental example of using BERT for classification. Seting `--clean_notes=True` and `--mode train` changes the texts into the required format and fine-tune the model.
```
python ModelsBert.py \
--clean_notes=True \
--mode train \
--path ../data/EPIC_DATA/EPIC.csv \
--task_name customized_task_name
```

## Model Evaluation
To test the fine-tuned model:
```
python ModelsBERT.py \
--clean_notes False \
--mode test \
--path /home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv \
--task_name customized_task_name
```