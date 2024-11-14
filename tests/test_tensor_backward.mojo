import testing

from mocrograd import tensor, grads


fn create_a() -> tensor.Tensor[2, 3, grads.NoneBackward]:
    data = tensor.Matrix[2, 3]()
    data[0, 0] = 1.0
    data[0, 1] = 2.0
    data[0, 2] = 3.0
    data[1, 0] = 4.0
    data[1, 1] = 5.0
    data[1, 2] = 6.0
    return tensor.Tensor[2, 3, grads.NoneBackward](data^, requires_grad=True)


fn create_b() -> tensor.Tensor[3, 1, grads.NoneBackward]:
    data = tensor.Matrix[3, 1]()
    data[0, 0] = 7.0
    data[1, 0] = 8.0
    data[2, 0] = 9.0
    return tensor.Tensor[3, 1, grads.NoneBackward](data^, requires_grad=True)


fn test_sum() raises -> None:
    m1 = create_a()
    result = m1.sum()

    testing.assert_equal(result.rows, 1)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result[0, 0], 21.0)


fn test_sum_back() raises -> None:
    m1 = create_a()
    result = m1.sum()

    testing.assert_equal(result.rows, 1)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result[0, 0], 21.0)

    result.backward()


fn test_matmul() raises -> None:
    m1 = create_a()
    m2 = create_b()
    result = m1 @ m2

    testing.assert_equal(result.rows, 2)
    testing.assert_equal(result.cols, 1)

    testing.assert_equal(result[0, 0], 50.0)
    testing.assert_equal(result[1, 0], 122.0)


fn run_tests() raises -> None:
    test_sum()
    test_matmul()

    print("All tensor backward pass tests passed!")
