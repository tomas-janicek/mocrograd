# Timed Training Test

## Digits Dataset - 100 Examples

Tested with: `magic run train_digits` with 1 epoch and length set to 100 for digits dataset in batches of 32.

### MacBook M1

MacBook M1 uses nelts with value 16.

#### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 1.9945100000000002 seconds
Training with full optimization (10 workers): 2.216365 seconds
Training with full optimization (64 workers): 2.7613130000000004 seconds
Training with full optimization (256 workers): 5.279564000000001 seconds

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 57.613562 seconds
Training with full optimization (10 workers): 58.414107 seconds
Training with full optimization (64 workers): 60.850778000000005 seconds
Training with full optimization (256 workers): 74.03118400000001 seconds

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 578.785515 seconds
Training with full optimization (10 workers): 652.545959 seconds
Training with full optimization (64 workers): 622.938336 seconds
Training with full optimization (256 workers): 634.539341 seconds

### Run Pod

NVidia RTX 4000 Ada uses nelts with value 16.

#### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 1.21092444 seconds
Training with full optimization (10 workers): 2.0891751860000003 seconds
Training with full optimization (64 workers): 4.941721467000001 seconds
Training with full optimization (256 workers): 12.003985395 seconds

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 36.638341556 seconds
Training with full optimization (10 workers): 65.684224473 seconds
Training with full optimization (64 workers): 84.700366343 seconds
Training with full optimization (256 workers): 139.984655003 second

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 398.27412804600004 seconds
Training with full optimization (10 workers): 341.15016408900004 seconds
Training with full optimization (64 workers): 346.273877286 seconds
Training with full optimization (256 workers): 367.12238372800005

## Digits Dataset - 10 Examples

Tested with: `magic run train_digits` with 1 epoch and length set to 10 for digits dataset in batches of 32.

### MacBook M1

MacBook M1 uses nelts with value 16.

#### Network size 64 -> 64 -> 32 -> 10

Python implementation: 0.11281895637512207 seconds

Training without any optimization: 0.01026 seconds
Training with full optimization (1 workers): 0.010337 seconds
Training with full optimization (10 workers): 0.011443 seconds
Training with full optimization (64 workers): 0.013864000000000001 seconds
Training with full optimization (256 workers): 0.018944000000000003 seconds
Training with dynamic workers number: 0.01124 seconds

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Python implementation: 38.23866581916809 seconds

Training without any optimization: 0.453761 seconds
Training with full optimization (1 workers): 0.455959 seconds
Training with full optimization (10 workers): 0.415574 seconds
Training with full optimization (64 workers): 0.44307100000000005 seconds
Training with full optimization (256 workers): 0.475797 seconds
Training with dynamic workers number: 0.486175 seconds

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Python implementation: 916.3234758377075 seconds

Training without any optimization: 5.679394 seconds
Training with full optimization (1 workers): 5.741174 seconds
Training with full optimization (10 workers): 4.1578040000000005 seconds
Training with full optimization (64 workers): 4.074032 seconds
Training with full optimization (256 workers): 4.079474 seconds
Training with dynamic workers number: 4.176993 seconds
