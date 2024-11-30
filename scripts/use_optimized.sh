#!/usr/bin/env bash

set -e

[ -e mocrograd/matrix_perf.mojo ] \
    && mv mocrograd/matrix.mojo mocrograd/matrix_origin.mojo \
    && mv mocrograd/matrix_perf.mojo mocrograd/matrix.mojo

[ -e mocrograd/grads_perf.mojo ] \
    && mv mocrograd/grads.mojo mocrograd/grads_origin.mojo \
    && mv mocrograd/grads_perf.mojo mocrograd/grads.mojo
