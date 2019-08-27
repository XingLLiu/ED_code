# Clinical BERT Intall and Usage Guide
This is a usage guide of fine-tuning the [Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT) ([E. Alsentzer *et al* 2019](https://arxiv.org/abs/1904.03323)) model for Sepsis prediction. 


## Recommended readings
1. The original BERT paper can be found [here](https://arxiv.org/abs/1810.04805).
2. For a gental introduction to how BERT works, please see [here](https://medium.com/synapse-dev/understanding-bert-transformer-attention-isnt-all-you-need-5839ebd396db) and [here](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73).
3. Click [here](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04) for a nice example of fine-tuning BERT on a classification problem using PyTorch.


## Folder structure
There is no need to manually create any folders when running `bert_pipeline.py`, as long as the folder structure described in the root directory is followed. 

By default, all outputs from `bert_pipeline.py` will be stored in `/ED/results/bert/` as follows:

```
|-- Raw_Notes                                 # Cleaned triage notes
|-- dynamic
    |-- task_name
        |-- month
            |-- Processed_Texts               # Train and test sets in the format required by BERT
            |-- Saved_Checkpoints             # Fine-tuned models
            |-- predicted_result_yyyymm.csv   # Predicted probability of the test set
            |-- predicted_result_train_yyyymm.csv   # Predicted probability of the train set
            |-- summary.csv                   # TR, NR, TPR, FPR of the prediction
            |__ roc_yyyymm                    # ROC plot
```

## Download Clinical BERT
Assuming the current directory is '/ED/ED_code/', the following code downloads the pre-trained [Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT) ([E. Alsentzer *et al* 2019](https://arxiv.org/abs/1904.03323)) and unzips it into the folder `/ED/ClinicalBert/`:
```
mkdir ../ClinicalBert
wget -O ../ClinicalBert/pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
tar zxvf ../ClinicalBert/pretrained_bert_tf.tar.gz -C ../ClinicalBert/
```

## Fine-Tune Clinical BERT
To fine-tune the Clinical BERT, change the current working directory to `/ED_code/Modelling` and run
```
sh run_bert_pipeline.sh
```
This will call `bert_pipeline.py`, which
1. fine-tunes the BERT model under the one-month ahead framework,
2. tars the fine-tuned model into `ED/results/bert/dynamic/epic_task/month/Saved_Checkpoints/` for later reference,
3. evaluates the performance and save the plots in `ED/results/bert/dynamic/epic_task/month/`,
4. repeats for every month.
