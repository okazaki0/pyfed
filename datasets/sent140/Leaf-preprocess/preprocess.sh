#!/usr/bin/env bash

# download data and convert to .json format
cd datasets/sent140/Leaf-preprocess

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi


NAME="sent140" # name of the dataset, equivalent to directory name

cd utils

./preprocess.sh --name $NAME $@

cd ..
if [ ! -f 'glove.6B.300d.txt' ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
fi

if [ ! -f embs.json ]; then
    python3 get_embs.py
fi