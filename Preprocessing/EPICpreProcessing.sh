# All preprocessing work to produce preprocessed_EPIC.csv
# To run: 
# 1. Change the current directory to /ED/ED_code/
# 2. On the command line: 
#    ./Preprocessing/EPICpreProcessing.sh

# Change the working directory to ./Preprocessing
cd ./Preprocessing

echo "----- Checking if required R packages are installed -----"
echo "If this returns an error, please make sure all R packages in requiredPackages.R is installed."
Rscript requiredPackages.R
echo -e "\nAll required packages loaded\n"

echo "----- Combining data sources to produce EPIC.csv -----"
Rscript combineEPICData.R
echo -e "Completed\n"

echo "----- Appedning RN and MD alerts to produce EPIC_with_alerts.csv -----"
python appendAlerts.py
echo -e "Completed\n"

echo "----- Preprocessing EPIC_with_alerts.csv to produce preprocessed_EPIC.csv -----"
Rscript preprocessEPIC.R
echo -e "Completed\n"