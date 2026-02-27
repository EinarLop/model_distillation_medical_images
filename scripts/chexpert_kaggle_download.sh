#!/bin/bash
set -e

DATA_DIR="data"

echo "Creating data directory..."
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Downloading and unzipping CheXpert Small from Kaggle..."

kaggle datasets download -d ashery/chexpert

echo "CheXpert is ready in the /$DATA_DIR directory."