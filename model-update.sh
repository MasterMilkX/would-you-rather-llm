#!/bin/bash

.venv/bin/python3 proc_moderate_data.py
.venv/bin/python3 finetuning-notebooks/train_all_models.py