## Description

Contains code for ED project. Assume that ED_Code is a subfolder of ~/ED, where ~/ED/ED_code, will execute on ~/ED/ED_data.

## Update after [16/07/2019]

Now assume that ED_code is contained in the same subfolder as the ED folder available from the PHF. To run the preprocessing scripts:
```bash  
Rscript ./Preprocessing/EPICscript.R
```

## Code structure
```bash
.
|-- Modelling 
|   |-- EDA.py                   # EDA and modelling on the preprocessed data
|   |__ EDA_old_data.py          # EDA on the orignial EPIC data
|-- Preprocessing
|   |-- ED_data_process.sh     
|   |-- EPICscript.R             # Source combineEPICData.R and preprocessEPIC.R
|   |-- combineEPICData.R        
|   |__ preprocessEPIC.R  
|-- ED_devin_data_clean_excel.py
|-- ED_support_funs.py
|-- README.md
|__ test.txt
```

