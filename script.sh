#!/bin/bash
# This is for auto-run

python3 main.py --model-name squeezenet1_1 --batch-size 128 --val-batch-size 200 | tee squeezenet1_1_128.txt
python3 main.py --model-name densenet121 --batch-size 128 --val-batch-size 200 | tee densenet121_128.txt
python3 main.py --model-name squeezenet1_1 --batch-size 128 --val-batch-size 200 | tee squeezenet1_1_64.txt
python3 main.py --model-name densenet121 --batch-size 64 --val-batch-size 200 | tee densenet121_64.txt