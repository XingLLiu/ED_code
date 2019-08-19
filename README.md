## Description

Contains code for ED project. Assume that ED_Code is a subfolder of ~/ED, where ~/ED/ED_code, will execute on ~/ED/ED_data.

## Update after [27/07/2019]

Now assume that ED_code is contained in the same subfolder as the ED folder available from the PHF. To run the preprocessing scripts:
```bash  
./Preprocessing/EPICpreProcessing.sh
```

## Code structure
```bash

|-- Archive                      # Archived files
|-- Modelling 
|   |-- EDA.py                   # EDA and modelling on the preprocessed data
|   |-- EDA_old_data.py          # EDA on the orignial EPIC data
|   |-- Models.py                # Modelling
|   |-- ED_support_module        # Functions used by other scripts
|       |__ __init__.py
|   |__ saved_results            # Saved results from the models
|-- Preprocessing
|   |-- ED_data_process.sh     
|   |-- combineEPICData.R        
|   |-- preprocessEPIC.R 
|   |-- EPICpreProcessing.sh     # Produce EPIC.csv
|   |-- appendAlerts.py          # Produce EPICpreProcessed.csv
|   |-- requiredPackages.R       # Packages required by EPICpreProcessing.sh
|   |__ Sepsis_Reports.R      
|-- Exploratory
|   |-- ED_devin_data_clean_excel.py
|   |-- ED_support_funs.py
|__ README.md
```


## Set-up virtual environment

Furture versions of packages might not be compatible with the current code and implementation, so it is important to set up a virtual environment before running the scripts. `ED_environment.yml` has the required python version and packages installed. To install:

1. Install the latest conda ([tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)) or update as:
```bash
conda update conda
```

2. Install the virtual environment:
```bash
conda env create -f ED_environment.yml
```

3. Enter the virtual environment as follows when working on the project:
```bash
source activate 
```


