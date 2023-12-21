#!/bin/sh
module purge
module load Python/3.11
source ts_venv/bin/activate

srun -c 16 --mem-per-cpu=2g  --output=output/bi_gru1.log python3 Bi_GRU.py

