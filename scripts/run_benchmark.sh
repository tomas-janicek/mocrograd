#!/usr/bin/env bash

set -e

bash scripts/use_original.sh

echo
echo "Training with Original Matrix Implementation"
echo "############################################"
magic run train_digits

bash scripts/use_optimized.sh

echo
echo "Training with Optimized Matrix Implementation"
echo "#############################################"

echo
echo "Train with 1 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 1
magic run train_digits

echo
echo "Train with 10 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 10
magic run train_digits

echo
echo "Train with 64 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 64
magic run train_digits

echo
echo "Train with 256 workers"
echo "----------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 256
magic run train_digits

bash scripts/use_dynamic_optimized.sh

echo
echo "Training with Dynamic Optimized Matrix Implementation"
echo "#####################################################"
magic run train_digits