# Timed Training Test

## Digits Dataset

Tested with: `magic run train_digits` with 20 epochs and length set to 1000 for digits dataset.

### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 38.716842 seconds
Training with matmul vectorized and parallelized: 39.672311 seconds
Training with matmul and matmul backwards vectorized and parallelized: 40.950168000000005 seconds

### Network size 64 -> 512 -> 256 -> 10

Training without any optimization: 82.484564 seconds
Training with full optimization: 103.813782 seconds
