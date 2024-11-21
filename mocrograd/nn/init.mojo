import math
import random

from mocrograd import tensor, matrix


fn create_normal_weights(rows: UInt, cols: UInt) -> tensor.Tensor:
    return tensor.Tensor.rand(rows, cols, requires_grad=True)


fn create_kaiming_normal_weighta(rows: UInt, cols: UInt) -> tensor.Tensor:
    var random_matrix = matrix.Matrix.randn(rows, cols)
    var data = random_matrix * (math.sqrt(Float32(2.0) / rows))
    return tensor.Tensor(data, requires_grad=True)
