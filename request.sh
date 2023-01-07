#!/usr/bin/env bash

python3 "$1"create_csv.py -t "request" >> /dev/null && python3 "$1"request.py
