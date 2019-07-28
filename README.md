## Description

Contains code for ED project. Assume that ED_Code is a subfolder of ~/ED, where ~/ED/ED_code, will execute on ~/ED/ED_data.

## Update after [27/07/2019]

Now assume that ED_code is contained in the same subfolder as the ED folder available from the PHF. To run the preprocessing scripts:
```bash  
Rscript ./Preprocessing/EPICpreProcessing.sh
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

