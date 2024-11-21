import testing

from memory import Arc

from mocrograd import tensor, grads, matrix


fn _create_tensor() -> tensor.Tensor:
    var data = matrix.Matrix(2, 3)
    data[0, 0] = 1.0
    data[0, 1] = 2.0
    data[0, 2] = 3.0
    data[1, 0] = 4.0
    data[1, 1] = 5.0
    data[1, 2] = 6.0
    return tensor.Tensor(data=data^, requires_grad=True)


fn test_get() raises -> None:
    var t = _create_tensor()

    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 2.0)
    testing.assert_equal(t[1, 1], 5.0)
    testing.assert_equal(t[1, 2], 6.0)


fn test_set() raises -> None:
    var t = tensor.Tensor(2, 3, requires_grad=True)
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
    var t1 = _create_tensor()
    testing.assert_equal(t1[1, 2], 6.0)

    var t2 = t1^
    testing.assert_equal(t2[1, 2], 6.0)
    # Cannot even access t1
    # testing.assert_equal(t1[1, 2], 6.0)


fn test_copy() raises -> None:
    var t1 = _create_tensor()
    testing.assert_equal(t1[1, 2], 6.0)

    var t2 = t1
    testing.assert_equal(t1[1, 2], 6.0)
    testing.assert_equal(t2[1, 2], 6.0)

    # Matrix is using same underlying data even when we copy the structure.
    t2[1, 2] = 7.0

    testing.assert_equal(t1[1, 2], 7.0)
    testing.assert_equal(t2[1, 2], 7.0)


fn run_tests() raises -> None:
    test_get()
    test_set()
    test_move()
    test_copy()

    print("All tensor tests passed!")
