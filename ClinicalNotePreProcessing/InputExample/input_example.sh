#!/bin/bash
# Assume current directory = /ED_code/ClinicalNotePreProcessing

dir_apache=~/Documents/Ctakes/apache-ctakes-4.0.0/bin
dir=$(pwd)
dir_eg=$(pwd)/InputExample

cd $dir_apache
$dir_apache/runClinicalPipeline.sh  -i $dir_eg/annotatorInput  --xmiOut $dir_eg/annotatorOutput   --user $uname  --pass $passwd
echo "----- Running python CUI extractor script -----"
cd $dir
python $dir/extract_CUIs.py --path $dir_EPIC