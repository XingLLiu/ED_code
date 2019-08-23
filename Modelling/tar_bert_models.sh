#!/bin/sh
# Tar the pre-trained BERT model for later reference

CODE_DIR=$(pwd)
echo "$(tput setaf 1)Zipping fine-tuned model ...$(tput sgr 0)"
# tar -czvf $1 $2 $3
cd $1
tar -czvf $2.tar.gz $3 $4
cd $CODE_DIR
echo "$(tput setaf 1)Complete.\n$(tput sgr 0)"

