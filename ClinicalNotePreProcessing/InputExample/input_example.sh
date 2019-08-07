#!/bin/bash
# Assume current directory = /ED_code/ClinicalNotePreProcessing
# Maria's user/pass
uname="MariaY"
passwd="Frat3llis!"

dir_apache=~/Documents/Ctakes/apache-ctakes-4.0.0/bin
dir=$(pwd)

cd $dir_apache
$dir_apache/runClinicalPipeline.sh  -i $dir/annotatorInput  --xmiOut $dir/annotatorOutput   --user $uname  --pass $passwd
