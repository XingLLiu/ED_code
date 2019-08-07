#!/bin/bash

dir_apache=~/Documents/Ctakes/apache-ctakes-4.0.0/bin
dir_

# Maria's user/pass
uname="MariaY"
passwd="Frat3llis!"

cd $dir_apache
$dir_apache/runClinicalPipeline.sh  -i $dir_EPIC/annotatorInput  --xmiOut $dir_EPIC/annotatorOutput   --user $uname  --pass $passwd
