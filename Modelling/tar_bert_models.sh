#!/bin/sh
# ---------------------------------------------------------------- #
# Tar the pre-trained BERT model for later reference.
# Called by bert_pipeline.py
# ---------------------------------------------------------------- #

# Directory of the bert pipeline code
CODE_DIR=$(pwd)

# Tar model files
echo "$(tput setaf 1)\nZipping fine-tuned model ...$(tput sgr 0)"
cd $1
tar -czvf $2.tar.gz $3 $4
cd $CODE_DIR
echo "$(tput setaf 1)Complete.\n$(tput sgr 0)"

