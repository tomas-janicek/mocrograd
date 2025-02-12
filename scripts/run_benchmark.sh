#!/usr/bin/env bash

set -e

epochs=10
length=1000

bash scripts/use_original.sh

echo
echo "Training with Original Matrix Implementation"
echo "############################################"
magic run mojo run run_digits_training.mojo $epochs $length

bash scripts/use_optimized.sh

echo
echo "Training with Optimized Matrix Implementation"
echo "#############################################"

echo
echo "Train with 1 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 1
magic run mojo run run_digits_training.mojo $epochs $length

echo
echo "Train with 10 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 10
magic run mojo run run_digits_training.mojo $epochs $length

echo
echo "Train with 64 workers"
echo "---------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 64
magic run mojo run run_digits_training.mojo $epochs $length

echo
echo "Train with 256 workers"
echo "----------------------"
bash scripts/set_num_workers.sh mocrograd/matrix.mojo 256
magic run mojo run run_digits_training.mojo $epochs $length

bash scripts/use_dynamic_optimized.sh

echo
echo "Training with Dynamic Optimized Matrix Implementation"
echo "#####################################################"
magic run mojo run run_digits_training.mojo $epochs $length