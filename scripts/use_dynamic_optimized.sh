#!/usr/bin/env bash

set -e

cp mocrograd/matrix_implementations/matrix_dynamic_workers.mojo mocrograd/matrix.mojo
cp mocrograd//matrix_implementations/grads_dynamic_workers.mojo mocrograd/grads.mojo
