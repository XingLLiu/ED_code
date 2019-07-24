#!/bin/bash

cd /
cd path/to/ED/ED_code
python cleaning_script.py
cd /
cd path/to/apache-ctakes-4.0.0/bin
/bin/bash path/to/apache-ctakes-4.0.0/bin/runClinicalPipeline  -i path/to/ED/ED_code/annotatorInput  --xmiOut path/to/ED/ED_code/annotatorOutput   --user UMLSUser  --pass UMLSpassword
cd /
cd path/to/ED/ED_code
python extract_CUIs.py