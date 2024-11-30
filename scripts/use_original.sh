#!/usr/bin/env bash

set -e

cp mocrograd/matrix_implementations/matrix_origin.mojo mocrograd/matrix.mojo
cp mocrograd//matrix_implementations/grads_origin.mojo mocrograd/grads.mojo
