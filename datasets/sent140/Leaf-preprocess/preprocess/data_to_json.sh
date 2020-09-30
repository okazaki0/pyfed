#!/usr/bin/env bash

if [ ! -d "../data" ]; then
  mkdir ../data
fi
if [ ! -d "../data/raw_data" ]; then
  mkdir ../data/raw_data
fi
if [ ! -f ../data/raw_data/test.csv ]; then
  echo "------------------------------"
  echo "retrieving raw data"
  
  ./get_data.sh
  echo "finished retrieving raw data"
fi

if [ ! -d "../data/intermediate" ]; then
  echo "------------------------------"
  echo "combining raw_data .csv files"
  mkdir ../data/intermediate
  python3 combine_data.py
  echo "finished combining raw_data .csv files"
fi

if [ ! -f ../data/all_data/test.json ]; then
  echo "------------------------------"
  echo "converting test data to .json format"
  mkdir ../data/all_data
  python3 data_to_json.py --file test
  echo "finished converting data to .json format"
fi
if [ ! -f ../data/all_data/training.json ]; then
  echo "------------------------------"
  echo "converting training data to .json format"
  python3 data_to_json.py --file training
  echo "finished converting data to .json format"
fi
