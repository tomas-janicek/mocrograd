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


fn test_get() raises -> None:
    t = create_a()

    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 2.0)
    testing.assert_equal(t[1, 1], 5.0)
    testing.assert_equal(t[1, 2], 6.0)


fn test_set() raises -> None:
    t = tensor.Tensor[2, 3, grads.NoneBackward](requires_grad=True)
    t[0, 0] = 1.0
    t[0, 1] = 2.0
    t[0, 2] = 3.0
    t[1, 0] = 4.0
    t[1, 1] = 5.0
    t[1, 2] = 6.0

    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 2.0)
    testing.assert_equal(t[1, 1], 5.0)
    testing.assert_equal(t[1, 2], 6.0)


fn test_move() raises:
    t1 = create_a()
    testing.assert_equal(t1[1, 2], 6.0)

    t2 = t1^
    testing.assert_equal(t2[1, 2], 6.0)
    # Cannot evem access t1
    # testing.assert_equal(t1[1, 2], 6.0)


fn test_copy() raises -> None:
    t1 = create_a()
    testing.assert_equal(t1[1, 2], 6.0)

    t2 = t1
    testing.assert_equal(t1[1, 2], 6.0)
    testing.assert_equal(t2[1, 2], 6.0)


fn run_tests() raises -> None:
    test_get()
    test_set()

    print("All tensor tests passed!")
