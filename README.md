# Mocrograd

Mocrograd is a lightweight library for automatic differentiation. It is designed to be simple and easy to use, making it a great choice for educational purposes and small projects.

It is closely related to [Pycrograd](https://github.com/TomasJani/pycrograd]. It is used to the same educational purposes with the same motivation (described below), but it is more optimized and uses less memory.

## 🎯 Motivation

The motivation behind building Mocrograd includes several key objectives:
- To gain a deeper understanding of how autograd and PyTorch work.
- To learn how to build a neural network from scratch.
- To experiment with GPU and memory optimization, which is planned for future implementation.
- To explore various optimization techniques such as vectorization, parallelization, tiling, and parameterization.
- To understand how to efficiently save and retrieve data from memory during model training.

## ✨ Features

- **Automatic Differentiation**: Compute gradients automatically for your models.
- **Neural Networks**: Build and train neural networks with ease.
- **Matrix operations**: Perform matrix operations like sum, log, etc.

## 🛠️ Implementations

### 📜 Original Implementation

The cornerstone of implementations is `Tensor`, `Matrix` class and set of gradient functions. The `Tensor` class is used to store the value and gradient of a node in the computational graph. `Tensor` user `Matrix` class to hold both value and gradient data. The `Matrix` class is used to perform matrix operations like sum, log, etc. The gradient functions are used to calculate the gradients of the operations performed on the tensors.

Compared to Pycrograd, Mocrograds `Matrix` class is more optimized and uses less memory. It work with memory directly using unsafe pointers. Because we know the size of the matrix, we can allocate memory for the matrix only once and reuse it. 

### ⚡ Optimized Implementation

Optimized impmenetetions are implemented in `mocrograd/matrix_implementations/grads_optimized.mojo` and `mocrograd/matrix_implementations/matrix_optimized.mojo` and work almost the same as original implementation.

The main difference is the use of vectorization and parallelization. The optimized implementation uses multiple workers to perform matrix operations in parallel. This significantly reduces the time taken to perform matrix operations, especially for large matrices. The number of workers can be adjusted to optimize performance based on the available resources.

### 🔄 Dynamic Optimized Implementation

Optimized implementations with dynamic number of workers are implemented in `mocrograd/matrix_implementations/grads_dynamic_workers.mojo` and `mocrograd/matrix_implementations/matrix_dynamic_workers.mojo`.

The dynamic optimized implementation uses a dynamic number of workers to perform matrix operations in parallel. The number of workers is adjusted based on the size of the matrix. It uses number of columns in the matrix to determine the number of elements used for vectorization and number of rows in the column to determine the number of parallelization workers. 

## 🚀 Next Steps

1) Make the code run on NVIDIA and AMD GPUs.

## 📦 Installation

To install the required dependencies, use the following command:

```sh
magic install
```

## 📘 Usage

In this case, 10 represent the number of epochs and 100 represent the number of examples in the dataset.

```sh
magic run mojo run run_digits_training.mojo 10 100
```

## 🧪 Running Tests

To run the tests, use the following command:

```sh
magic run mojo run run_tests.mojo
```

## 📊 Benchmarks

To run the benchmarks (it uses 10 epochs and 100 examples in dataset by default), use the following command:

```sh
bash scripts/run_benchmark.sh
```

The `run_benchmark.sh` script performs the following tasks:

1. Trains a model using the original implementation.
2. Switches to the optimized implementation and trains the model with different numbers of workers (1, 10, 64, 256).
3. Switches to the dynamic optimized implementation and trains the model again.

This helps to compare the performance of different implementations and configurations.

### 📈 Benchmarks Results

All benchmarks are run on a MacBook Pro M1 with 32 GB of RAM. MacBook Pro M1 uses nelts with value 16.

#### Network size **[Normal]** 64 -> 64 -> 32 -> 10

Training without any optimization: 20.281 seconds
Training with optimization (1 worker): 20.587 seconds
Training with optimization (10 workers): 20.907 seconds
Training with optimization (64 workers): 23.513 seconds
Training with optimization (256 workers): 30.205 seconds
Training with optimization (dynamic workers): 21.602 seconds

#### Network size **[Longer]** 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 56.842 seconds
Training with optimization (1 worker): 56.849 seconds
Training with optimization (10 workers): 55.059 seconds
Training with optimization (64 workers): 56.790 seconds
Training with optimization (256 workers): 59.224 seconds
Training with optimization (dynamic workers): 58.162 seconds

#### Network size **[Bigger]** 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 94.953 seconds
Training with optimization (1 worker): 92.970 seconds
Training with optimization (10 workers): 78.058 seconds
Training with optimization (64 workers): 80.794 seconds
Training with optimization (256 workers): 79.919 seconds
Training with optimization (dynamic workers): 87.143 seconds

# 📚 Sources

- [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
- [A Simple Introduction to Cross Entropy Loss](https://insidelearningmachines.com/cross_entropy_loss/)
- [Backpropagation through softmax layer](https://binpord.github.io/2021/09/26/softmax_backprop.html)
