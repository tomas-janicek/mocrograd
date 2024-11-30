from collections import Optional
from algorithm import parallelize, vectorize

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

    @parameter
    fn _sum_backward_row(row: Int):
        @parameter
        fn _sum_backward[nelts: Int](col: Int):
            # left.grad[row, col] += out_grad
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col) + out_grad,
            )

        vectorize[_sum_backward, matrix.nelts](size=left.cols)

    parallelize[_sum_backward_row](left.rows, matrix.num_workers)


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

    @parameter
    fn _addition_backward_row(row: Int):
        @parameter
        fn _addition_backward[nelts: Int](col: Int):
            if left.grad:
                # left.grad[row, col] += grad[row, col]
                left.grad.value().store[nelts](
                    row,
                    col,
                    left.grad.value().load[nelts](row, col)
                    + grad.value().load[nelts](row, col),
                )
            if right.grad:
                # right.grad[row, col] += grad[row, col]
                right.grad.value().store[nelts](
                    row,
                    col,
                    right.grad.value().load[nelts](row, col)
                    + grad.value().load[nelts](row, col),
                )

        vectorize[_addition_backward, matrix.nelts](size=left.cols)

    parallelize[_addition_backward_row](left.rows, matrix.num_workers)


fn addition_number_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    @parameter
    fn _addition_backward_row(row: Int):
        @parameter
        fn _addition_backward[nelts: Int](col: Int):
            # left.grad[row, col] += grad[row, col]
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col)
                + grad.value().load[nelts](row, col),
            )

        vectorize[_addition_backward, matrix.nelts](size=left.cols)

    parallelize[_addition_backward_row](left.rows, matrix.num_workers)


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

    @parameter
    fn _matmul_backward_row(m: Int):
        for k in range(left.cols):

            @parameter
            fn _matmul_backward[nelts: Int](n: Int):
                if left.grad:
                    # left.grad[m, k] += right[k, n] * grad[m, n]
                    left.grad.value().store[nelts](
                        m,
                        k,
                        left.grad.value().load[nelts](m, k)
                        + right[k, n] * grad.value().load[nelts](m, n),
                    )
                if right.grad:
                    # right.grad[k, n] += left[m, k] * grad[m, n]
                    right.grad.value().store[nelts](
                        k,
                        n,
                        right.grad.value().load[nelts](k, n)
                        + left[m, k] * grad.value().load[nelts](m, n),
                    )

            vectorize[_matmul_backward, matrix.nelts](size=out.cols)

    parallelize[_matmul_backward_row](out.rows, matrix.num_workers)


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

    @parameter
    fn _power_backward_row(row: Int):
        @parameter
        fn _power_backward[nelts: Int](col: Int):
            # left.grad[row, col] += (power * left[row, col] ** (power - 1)) * grad[row, col]
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col)
                + (power * left.data.load[nelts](row, col) ** (power - 1))
                * grad.value().load[nelts](row, col),
            )

        vectorize[_power_backward, matrix.nelts](size=left.cols)

    parallelize[_power_backward_row](left.rows, matrix.num_workers)


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

    @parameter
    fn _mul_backward_row(row: Int):
        @parameter
        fn _mul_backward[nelts: Int](col: Int):
            # left.grad[row, col] += multiplier * grad[row, col]
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col)
                + multiplier * grad.value().load[nelts](row, col),
            )

        vectorize[_mul_backward, matrix.nelts](size=left.cols)

    parallelize[_mul_backward_row](left.rows, matrix.num_workers)


fn relu_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previous: List[tensor.Tensor],
    grad_args: List[Float32],
) raises -> None:
    var left = previous[0]

    if not grad or not left.grad:
        raise "MissingGradError"

    @parameter
    fn _relu_backward_row(row: Int):
        @parameter
        fn _relu_backward[nelts: Int](col: Int):
            var relu_multiplier = 1 if out[row, col] > 0 else 0
            # left.grad[row, col] += relu_multiplier * grad[row, col]
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col)
                + relu_multiplier * grad.value().load[nelts](row, col),
            )

        vectorize[_relu_backward, matrix.nelts](size=left.cols)

    parallelize[_relu_backward_row](left.rows, matrix.num_workers)


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

    @parameter
    fn _log_softmax_backward_row(row: Int):
        @parameter
        fn _log_softmax_backward[nelts: Int](col: Int):
            # left.grad[row, col] += grad[row, col]
            left.grad.value().store[nelts](
                row,
                col,
                left.grad.value().load[nelts](row, col)
                + new_grad.load[nelts](row, col),
            )

        vectorize[_log_softmax_backward, matrix.nelts](size=left.cols)

    parallelize[_log_softmax_backward_row](left.rows, matrix.num_workers)
