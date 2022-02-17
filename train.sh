#!/usr/bin/env bash

# shellcheck disable=SC2035
touch 1.csv 1.dgl 1.pth 1.txt \
  && rm *.csv *.dgl *.pth *.txt \
  && python3 create_csv.py \
  && python3 make_dataset.py \
  && python3 train.py
