# Timed Training Test

## Digits Dataset

Tested with: `magic run train_digits` with 1 epoch and length set to 1000 for digits dataset in batches of 32.

### MacBook M1

MacBook M1 uses nelts with value 16.

#### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 1.9945100000000002 seconds
Training with full optimization (10 workers): 2.216365 seconds
Training with full optimization (64 workers): 2.7613130000000004 seconds
Training with full optimization (256 workers): 5.279564000000001 seconds

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 578.785515 seconds
Training with full optimization (10 workers): 652.545959 seconds
Training with full optimization (64 workers): 622.938336 seconds
Training with full optimization (256 workers): 634.539341 seconds

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 57.613562 seconds
Training with full optimization (10 workers): 65.0267 seconds
Training with full optimization (64 workers): 67.23092600000001 seconds
Training with full optimization (256 workers): 82.234509 seconds

### Run Pod

#### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 
Training with full optimization (10 workers): 
Training with full optimization (64 workers): 
Training with full optimization (256 workers): 

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 
Training with full optimization (10 workers): 
Training with full optimization (64 workers): 
Training with full optimization (256 workers): 

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 
Training with full optimization (10 workers): 
Training with full optimization (64 workers): 
Training with full optimization (256 workers): 
