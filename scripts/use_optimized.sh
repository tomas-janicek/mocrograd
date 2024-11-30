#!/usr/bin/env bash

set -e

cp mocrograd/matrix_implementations/matrix_optimized.mojo mocrograd/matrix.mojo
cp mocrograd//matrix_implementations/grads_optimized.mojo mocrograd/grads.mojo
