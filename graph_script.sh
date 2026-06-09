#!/bin/bash

.venv/bin/python3 proc_moderate_data.py
cd analysis
../.venv/bin/python3 data_analysis_code.py
cd ..