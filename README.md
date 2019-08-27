## Description

Contains code for ED project. Assume that ED_Code is a subfolder of ~/ED, where ~/ED/ED_code, will execute on ~/ED/ED_data.

## Update after [25/08/2019]

Now assume that ED_code is contained in the same subfolder as the ED folder available from the HPF. To run the preprocessing scripts:
```bash  
./Preprocessing/EPICpreProcessing.sh
```

## Code structure
```bash

|-- Archive                      # Archived files
|-- Modelling                    # Modelling pipelines
|   |__ ED_support_module        # Functions, modules and utils
|-- Preprocessing
|   |-- ED_data_process.sh     
|   |-- combineEPICData.R        
|   |-- preprocessEPIC.R 
|   |-- EPICpreProcessing.sh     # Creates EPIC.csv
|   |-- appendAlerts.py          # Creates EPICpreProcessed.csv
|   |-- requiredPackages.R       # Packages required by EPICpreProcessing.sh
|   |__ Sepsis_Reports.R
|-- Exploratory
|   |-- ED_devin_data_clean_excel.py
|   |-- ED_support_funs.py
|-- ED_environment.yml           # Python environment
|__ README.md
```


## Seting up virtual environment

Furture versions of packages might not be compatible with the current code and implementation, so it is important to set up a virtual environment before running the scripts. `ED_environment.yml` has the required python version and packages installed. To install:

1. Install the latest conda ([tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)) or update as:
```bash
conda update conda
```

2. Install the virtual environment:
```bash
conda env create -f ED_environment.yml
```

3. Activate the virtual environment as follows when working on the project:
```bash
source activate SepsisPrediction
```

4. You shoud then see the prefix of your terminal command line changed into the following:
```bash
(SepsisPrediction) username@pcname:~/path_to_ED/$
```


## Creating the required datasets
Multiple versions of the EPIC datasets are required for modelling. To create them,

1. Make sure you follow the folder structure as outlined at the top of this document.

2. Download CTakes and convert the clinical notes into CUIs as decribed in `./ClinicalNotePreProcessing/README.md`. This will take around 2 - 2.5 hours.

3. Run the following to create the required datasets.
```bash
sh ./Preprocessing/EPICpreProcessing.sh
```

## Modelling
An one-month ahead framework is used to train and test the models. A model is first trained on the first three months of data, tested on the fourth, and re-trained on the four months of data, tested on the fifth and so on. The motivation is that this would fit the real implementation setting better than random sampling for train/test splitting.

### Runing the existing models
Each script in the `./Modelling` ending with `pipeline.py` is the main script for that model. To run:

1. Customize the hyper-parameters in Section 0 of the script.

2. In Terminal, run
```bash
python some_model_pipeline.py
```

This will create a folder in `ED/results/model_name/` that contains all evaluation plots and prediction results.

The Python classes for the models, together with utility functions, can be found in `./Modelling/ED_support_module/`

### Customizing your own model
If you want to customize your own model and test it with the one-month ahead framework, you could copy a `pipeline.py` script and change

1. the beginning of the script to customize the hyperparameters,
2. all sections beginning with `2.a` in the script, as they are model-specific.

### Predicting other diseases
The current models predicts _Sepsis_ by default. The Python class `EPICPreprocess` does all downstream cleaning works, which binarize the response variable _Primary.Dx_ into binary classes. To run the same pipelines for other disease, change the argument `disease` when instantiating this class.

E.g. to predict for _abdominal pain_ instead, change section 1 of the scripts as follows
```python
# ========= 1. Further preprocessing =========
# List of strings of names of the disease of interest. Names are case-sensitive.
disease_lst = ["Abdominal pain", "abdominal pain"]
preprocessor = EPICPreprocess.Preprocess(DATA_PATH, disease = disease_lst)

# Run the preprocessing streamline and get multiple versions of the dataset
EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival = preprocessor.streamline()
```

The response variable _Primary.Dx_ will be converted to 1 if any of the disease names present in any of the features _Primary.Dx_, _Diagnosis_ and _Diagnoses_, and 0 otherwise. The rest of the code can be run as previous. 


