import testing
from collections import Set
from python import Python, PythonObject

from mocrograd import tensor, grads, matrix


fn _create_left() -> tensor.Tensor:
    var data = matrix.Matrix(2, 3)
    data[0, 0] = 1.0
    data[0, 1] = 2.0
    data[0, 2] = 3.0
    data[1, 0] = 4.0
    data[1, 1] = 5.0
    data[1, 2] = 6.0
    return tensor.Tensor(data^, requires_grad=True)


fn _create_right() -> tensor.Tensor:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = 7.0
    data[1, 0] = 8.0
    data[2, 0] = 9.0
    return tensor.Tensor(data^, requires_grad=True)


fn _create_relu() -> tensor.Tensor:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = -1.0
    data[1, 0] = 0.0
    data[2, 0] = 9.0
    return tensor.Tensor(data^, requires_grad=True)


fn test_grad_initialization() raises:
    var t_grad = tensor.Tensor.rand(2, 3, requires_grad=True)
    var t_no_grad = tensor.Tensor.rand(3, 2, requires_grad=False)

    testing.assert_true(t_grad.grad)
    testing.assert_false(t_no_grad.grad)

    var grad_value = t_grad.grad.value()
    testing.assert_equal(grad_value.rows, 2)
    testing.assert_equal(grad_value.cols, 3)

    testing.assert_equal(grad_value[0, 0], 0.0)
    testing.assert_equal(grad_value[1, 2], 0.0)


fn test_sum() raises:
    var left = _create_left()
    var result = left.sum()

    testing.assert_equal(result.rows, 1)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result.item(), 21.0)


fn test_sum_back() raises:
    var left = _create_left()
    var result = left.sum()

    result.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 1], 1.0)


fn test_matmul() raises:
    var left = _create_left()
    var right = _create_right()
    var result = left @ right

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result[0, 0], 50.0)
    testing.assert_equal(result[1, 0], 122.0)


fn test_matmul_backward() raises:
    var left = _create_left()
    var right = _create_right()
    var result = left @ right

    var sum = result.sum()

    testing.assert_equal(sum.item(), 172.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 7.0)
    testing.assert_equal(left.grad.value()[0, 1], 8.0)
    testing.assert_equal(left.grad.value()[1, 2], 9.0)

    testing.assert_true(right.grad)
    testing.assert_equal(right.grad.value()[2, 0], 9.0)
    testing.assert_equal(right.grad.value()[1, 0], 7.0)
    testing.assert_equal(right.grad.value()[0, 0], 5.0)


fn test_addition() raises:
    var left = _create_left()
    var right = _create_left()
    var result = left + right

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 2.0)
    testing.assert_equal(result[1, 0], 8.0)
    testing.assert_equal(result[1, 2], 12.0)


fn test_addition_backward() raises:
    var left = _create_left()
    var right = _create_left()
    var result = left + right

    var sum = result.sum()

    testing.assert_equal(sum.item(), 42.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 2], 1.0)

    testing.assert_true(right.grad)
    testing.assert_equal(right.grad.value()[0, 0], 1.0)
    testing.assert_equal(right.grad.value()[0, 1], 1.0)
    testing.assert_equal(right.grad.value()[1, 2], 1.0)


fn test_addition_number() raises:
    var left = _create_left()
    var result = left + 12.2

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 13.2)
    testing.assert_equal(result[1, 0], 16.2)
    testing.assert_equal(result[1, 2], 18.2)


fn test_addition_number_backward() raises:
    var left = _create_left()
    var result = left + 12.2

    var sum = result.sum()

    testing.assert_equal(sum.item(), 94.2)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 2], 1.0)


fn test_raddition_number() raises:
    var left = _create_left()
    var result = 12.2 + left

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 13.2)
    testing.assert_equal(result[1, 0], 16.2)
    testing.assert_equal(result[1, 2], 18.2)


fn test_raddition_number_backward() raises:
    var left = _create_left()
    var result = 12.2 + left

    var sum = result.sum()

    testing.assert_equal(sum.item(), 94.2)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 2], 1.0)


fn test_neg() raises:
    var left = _create_left()
    var result = -left

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], -1.0)
    testing.assert_equal(result[1, 0], -4.0)
    testing.assert_equal(result[1, 2], -6.0)


fn test_neg_backward() raises:
    var left = _create_left()
    var result = -left

    var sum = result.sum()

    testing.assert_equal(sum.item(), -21.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], -1.0)
    testing.assert_equal(left.grad.value()[0, 1], -1.0)
    testing.assert_equal(left.grad.value()[1, 2], -1.0)


fn test_sub() raises:
    var left = _create_left()
    var right = _create_left()
    var result = left - right

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 0.0)
    testing.assert_equal(result[1, 0], 0.0)
    testing.assert_equal(result[1, 2], 0.0)


fn test_sub_backward() raises:
    var left = _create_left()
    var right = _create_left()
    var result = left - right

    var sum = result.sum()

    testing.assert_equal(sum.item(), 0.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 2], 1.0)

    testing.assert_true(right.grad)
    testing.assert_equal(right.grad.value()[0, 0], -1.0)
    testing.assert_equal(right.grad.value()[0, 1], -1.0)
    testing.assert_equal(right.grad.value()[1, 2], -1.0)


fn test_sub_number() raises:
    var left = _create_left()
    var result = left - 12.2

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], -11.2)
    testing.assert_equal(result[1, 0], -8.2)
    testing.assert_equal(result[1, 2], -6.2)


fn test_sub_number_backward() raises:
    var left = _create_left()
    var result = left - 12.2

    var sum = result.sum()

    testing.assert_equal(sum.item(), -52.2)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 1.0)
    testing.assert_equal(left.grad.value()[0, 1], 1.0)
    testing.assert_equal(left.grad.value()[1, 2], 1.0)


fn test_rsub_number() raises:
    var left = _create_left()
    var result = 12.2 - left

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 11.2)
    testing.assert_equal(result[1, 0], 8.2)
    testing.assert_equal(result[1, 2], 6.2)


fn test_rsub_number_backward() raises:
    var left = _create_left()
    var result = 12.2 - left

    var sum = result.sum()

    testing.assert_equal(sum.item(), 52.2)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], -1.0)
    testing.assert_equal(left.grad.value()[0, 1], -1.0)
    testing.assert_equal(left.grad.value()[1, 2], -1.0)


fn test_mul() raises:
    var left = _create_left()
    var result = left * 6.0

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 6.0)
    testing.assert_equal(result[1, 0], 24.0)
    testing.assert_equal(result[1, 2], 36.0)


fn test_mul_backward() raises:
    var left = _create_left()
    var result = left * 6.0

    var sum = result.sum()

    testing.assert_equal(sum.item(), 126.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 6.0)
    testing.assert_equal(left.grad.value()[0, 1], 6.0)
    testing.assert_equal(left.grad.value()[1, 2], 6.0)


fn test_rmul() raises:
    var left = _create_left()
    var result = 6.0 * left

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 6.0)
    testing.assert_equal(result[1, 0], 24.0)
    testing.assert_equal(result[1, 2], 36.0)


fn test_rmul_backward() raises:
    var left = _create_left()
    var result = 6.0 * left

    var sum = result.sum()

    testing.assert_equal(sum.item(), 126.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 6.0)
    testing.assert_equal(left.grad.value()[0, 1], 6.0)
    testing.assert_equal(left.grad.value()[1, 2], 6.0)


fn test_pow() raises:
    var left = _create_left()
    var result = left**3

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 1.0)
    testing.assert_equal(result[1, 0], 64.0)
    testing.assert_equal(result[1, 2], 216.0)


fn test_pow_backward() raises:
    var left = _create_left()
    var result = left**3

    var sum = result.sum()

    testing.assert_equal(sum.item(), 441.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 3.0)
    testing.assert_equal(left.grad.value()[0, 1], 12.0)
    testing.assert_equal(left.grad.value()[1, 2], 108.0)


fn test_div_number() raises:
    var left = _create_left()
    var result = left / 4

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 3)

    testing.assert_equal(result[0, 0], 0.25)
    testing.assert_equal(result[1, 0], 1.0)
    testing.assert_equal(result[1, 2], 1.5)


fn test_div_number_backward() raises:
    var left = _create_left()
    var result = left / 4

    var sum = result.sum()

    testing.assert_equal(sum.item(), 5.25)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 0.25)
    testing.assert_equal(left.grad.value()[0, 1], 0.25)
    testing.assert_equal(left.grad.value()[1, 2], 0.25)


fn test_relu() raises:
    var left = _create_relu()
    var result = left.relu()

    testing.assert_equal(result.rows, 3)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result[0, 0], 0.0)
    testing.assert_equal(result[1, 0], 0.0)
    testing.assert_equal(result[2, 0], 9.0)


fn test_relu_backward() raises:
    var left = _create_relu()
    var result = left.relu()

    var sum = result.sum()

    testing.assert_equal(sum.item(), 9.0)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_equal(left.grad.value()[0, 0], 0.0)
    testing.assert_equal(left.grad.value()[1, 0], 0.0)
    testing.assert_equal(left.grad.value()[2, 0], 1.0)


fn test_log_softmax() raises:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = 0.5
    data[1, 0] = 0.75
    data[2, 0] = 0.01
    var left = tensor.Tensor(data^, requires_grad=True)
    var result = left.log_softmax()

    testing.assert_equal(result.rows, 3)
    testing.assert_equal(result.cols, 1)

    testing.assert_almost_equal(result[0, 0], -1.0636, atol=1e-4)
    testing.assert_almost_equal(result[1, 0], -0.8136, atol=1e-4)
    testing.assert_almost_equal(result[2, 0], -1.5536, atol=1e-4)


fn test_log_softmax_backward() raises:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = 0.5
    data[1, 0] = 0.75
    data[2, 0] = 0.01
    var left = tensor.Tensor(data^, requires_grad=True)
    var result = left.log_softmax()

    var sum = result.sum()

    testing.assert_almost_equal(sum.item(), -3.4307, atol=1e-4)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_almost_equal(left.grad.value()[0, 0], -0.0357, atol=1e-4)
    testing.assert_almost_equal(left.grad.value()[1, 0], -0.3298, atol=1e-4)
    testing.assert_almost_equal(left.grad.value()[2, 0], 0.3655, atol=1e-4)


fn test_log_softmax_one_big() raises:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = 0.5
    data[1, 0] = 200.0
    data[2, 0] = 2000.0
    var left = tensor.Tensor(data^, requires_grad=True)
    var result = left.log_softmax()

    testing.assert_equal(result.rows, 3)
    testing.assert_equal(result.cols, 1)

    testing.assert_almost_equal(result[0, 0], -1999.5, atol=1e-4)
    testing.assert_almost_equal(result[1, 0], -1800.0, atol=1e-4)
    testing.assert_almost_equal(result[2, 0], 0.0, atol=1e-4)


fn test_log_softmax_one_big_backward() raises:
    var data = matrix.Matrix(3, 1)
    data[0, 0] = 0.5
    data[1, 0] = 200.0
    data[2, 0] = 2000.0
    var left = tensor.Tensor(data^, requires_grad=True)
    var result = left.log_softmax()

    var sum = result.sum()

    testing.assert_almost_equal(sum.item(), -3799.5, atol=1e-4)

    sum.backward()

    testing.assert_true(left.grad)
    testing.assert_almost_equal(left.grad.value()[0, 0], 1.0, atol=1e-4)
    testing.assert_almost_equal(left.grad.value()[1, 0], 1.0, atol=1e-4)
    testing.assert_almost_equal(left.grad.value()[2, 0], -2.0, atol=1e-4)


fn run_tests() raises:
    test_grad_initialization()
    test_sum()
    test_sum_back()
    test_matmul()
    test_matmul_backward()
    test_addition()
    test_addition_backward()
    test_addition_number()
    test_addition_number_backward()
    test_raddition_number()
    test_raddition_number_backward()
    test_neg()
    test_neg_backward()
    test_sub()
    test_sub_backward()
    test_sub_number()
    test_sub_number_backward()
    test_rsub_number()
    test_rsub_number_backward()
    test_mul()
    test_mul_backward()
    test_rmul()
    test_rmul_backward()
    test_pow()
    test_pow_backward()
    test_div_number()
    test_div_number_backward()
    test_relu()
    test_relu_backward()
    test_log_softmax()
    test_log_softmax_backward()
    test_log_softmax_one_big()
    test_log_softmax_one_big_backward()

    print("All tensor backward pass tests passed!")
