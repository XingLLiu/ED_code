#!/bin/bash

# ----------- MASTER SHELL FOR ED DATA PROCESSING ----------- #

# ASSUMES THAT FOLDER STRUCTURE IS AS FOLLOWS:
# 	~/ED
# 	---- epic_all_data.csv
# 	---- ED_EPIC_DATA
# 	---- ED_code [GitLab]
# 	---- ---- ED_data_process.sh

# Get the base directory
dir_base=$(pwd)/../..
dir_code=$dir_base/ED_code/Preprocessing
dir_output=$dir_base/ED/data/EPIC_DATA/ED_EPIC_DATA
dir_data=$dir_output/ED_EPIC_DATA_JUNE2018_JUNE24_2019

# Comment dir_output and dir_data and comment out the following 
# two lines to process the Epic_Alert_Data
# dir_output=$dir_base/Epic_Sepsis_Alerts/Alerts_Data
# dir_data=$dir_output/../

# Make sure the directory structure is appropriate
echo "$(tput setaf 1)Contents of base directory:$(tput sgr 0)"
ls $dir_base
echo "$(tput setaf 1)Contents of code directory$(tput sgr 0)"
ls $dir_code
echo "$(tput setaf 1)Contents of data directory$(tput sgr 0)"
ls $dir_data

read -p "Are the folder contents correct (y/n)?" choice
if [[ $choice == "y" ]]; then
	echo "you choice yes... continuing script"
elif [[ $choice == "n" ]]; then
	echo "you choice no... breaking script"
	return
else
	echo "not a valid coice... breaking script"
	return
fi

# # ---- (0) DOWNLOAD DATA FROM HPF ---- #
# rm -r $dir_output
# scp -r edrysdale@data.ccm.sickkids.ca:/hpf/largeprojects/agoldenb/lauren/ED/ED_EPIC_DATA/ $dir_base

# ---- (1) PROCESS THE EXCEL FILES ---- #

python3 $dir_code/ED_devin_data_clean_excel.py --dir_data $dir_data --dir_output $dir_output --dir_code $dir_code

echo "end of the script!"


