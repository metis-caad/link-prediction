#!/usr/bin/env bash

mkdir -p requests/paths \
  && mkdir -p requests/connectivity/{1..5}_open \
  && mkdir -p requests/connectivity/{1..5}_closed \
  && mkdir -p requests/dataset_clean/{1..5}_open \
  && mkdir -p requests/dataset_clean/{1..5}_closed \
  && python3 server/lpapi/manage.py runserver 8001
