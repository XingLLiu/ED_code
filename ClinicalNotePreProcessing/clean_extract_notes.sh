#!/bin/bash

dir_apache=${1:-/usr/local/apache-ctakes-4.0.0/bin}

# Maria's user/pass
uname="MariaY"
passwd="Frat3llis!"

echo "----- Apache folder: "$dir_apache" -----"
# Check that runClinicalPipeline exists
apache_hit=$(ls $dir_apache | grep runClinicalPipeline.sh | wc -m)
if [ $apache_hit -lt 5 ]; then
	echo "Error! runClinicalPipeline.sh was not found in dir_apache"
fi

# Get the current diectory 
dir=$(pwd)
# Folder that should have the EPIC.csv file in it
dir_EPIC=$dir/../../data/EPIC_DATA
epic_hit=$(ls $dir_EPIC | grep ^EPIC.csv$ | wc -m)
if [ $epic_hit -lt 5 ]; then
	echo "Error! EPIC.csv was not found two folders up in ~data/"
	return
fi

echo "----- Running python cleaning script -----"
python $dir/cleaning_script.py --path $dir_EPIC --local $dir
echo "----- Changing folder to apache-ctakes -----"
cd $dir_apache
. $dir_apache/runClinicalPipeline.sh  -i $dir_EPIC/annotatorInput  --xmiOut $dir_EPIC/annotatorOutput   --user $uname  --pass $passwd
echo "----- Running python CUI extractor script -----"
cd $dir
python $dir/extract_CUIs.py --path $dir_EPIC
