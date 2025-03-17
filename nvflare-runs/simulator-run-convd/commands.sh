#!/bin/bash

# make sure to move smaller_dataset.csv into nvflare-runs/simulator-run-convd

docker build . -t misohu/nvflare:1.0

docker run -p 8888:8888 -v "$(pwd)":/workspace --gpus all misohu/nvflare:1.0