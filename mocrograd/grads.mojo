from collections import Optional

from mocrograd import tensor, matrix


fn sum_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    var out_grad = grad.value()[0, 0]
    for row in range(left.rows):
        for col in range(left.cols):
            left.grad.value()[row, col] += out_grad


fn addition_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]
    var right = previous[1]

    if not grad:
        raise "MissingGradError"
    if not left.grad and not right.grad:
        raise "MissingGradError"

    for row in range(left.rows):
        for col in range(left.cols):
            var grad_value = grad.value()[row, col]
            if left.grad:
                left.grad.value()[row, col] += grad_value
            if right.grad:
                right.grad.value()[row, col] += grad_value


fn addition_number_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    for row in range(left.rows):
        for col in range(left.cols):
            var grad_value = grad.value()[row, col]
            left.grad.value()[row, col] += grad_value


fn matmul_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]
    var right = previous[1]

    if not grad:
        raise "MissingGradError"
    if not left.grad and not right.grad:
        raise "MissingGradError"

    for m in range(left.rows):
        for k in range(left.cols):
            for n in range(right.cols):
                if left.grad:
                    left.grad.value()[m, k] += right[k, n] * grad.value()[m, n]
                if right.grad:
                    right.grad.value()[k, n] += left[m, k] * grad.value()[m, n]


fn power_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]
    var power = grad_args[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    for row in range(left.rows):
        for col in range(left.cols):
            left.grad.value()[row, col] += (
                power * left[row, col] ** (power - 1)
            ) * grad.value()[row, col]


fn mul_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]
    var multiplier = grad_args[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    for row in range(left.rows):
        for col in range(left.cols):
            left.grad.value()[row, col] += multiplier * grad.value()[row, col]


fn relu_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    for row in range(left.rows):
        for col in range(left.cols):
            var relu_multiplier = 1 if out[row, col] > 0 else 0
            left.grad.value()[row, col] += (
                relu_multiplier * grad.value()[row, col]
            )


fn log_softmax_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    new_grad = grad.value() - (out.exp() * grad.value().sum().item())
    for row in range(left.rows):
        for col in range(left.cols):
            left.grad.value()[row, col] += new_grad[row, col]
