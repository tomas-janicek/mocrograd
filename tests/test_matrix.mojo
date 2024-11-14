import testing

from mocrograd import tensor, grads


fn create_m() -> tensor.Matrix[2, 3]:
    data = tensor.Matrix[2, 3]()
    data[0, 0] = 1.0
    data[0, 1] = 2.0
    data[0, 2] = 3.0
    data[1, 0] = 4.0
    data[1, 1] = 5.0
    data[1, 2] = 6.0
    return data


fn test_get_set() raises:
    m = create_m()

    testing.assert_equal(m[0, 0], 1.0)
    testing.assert_equal(m[0, 2], 3.0)
    testing.assert_equal(m[1, 1], 5.0)
    testing.assert_equal(m[1, 2], 6.0)


fn test_move() raises:
    m1 = create_m()
    testing.assert_equal(m1[1, 2], 6.0)

    m2 = m1^
    testing.assert_equal(m2[1, 2], 6.0)
    # Cannot evem access m1
    # testing.assert_equal(m1[1, 2], 6.0)


fn test_copy() raises:
    m1 = create_m()
    testing.assert_equal(m1[1, 2], 6.0)

    m2 = m1
    testing.assert_equal(m1[1, 2], 6.0)
    testing.assert_equal(m2[1, 2], 6.0)


fn test_rand() raises -> None:
    t = tensor.Matrix[3, 3].rand()

    testing.assert_not_equal(t[0, 0], 0.0)
    testing.assert_not_equal(t[0, 1], 0.0)
    testing.assert_not_equal(t[2, 2], 0.0)


fn run_tests() raises -> None:
    test_get_set()
    test_move()
    test_copy()
    test_rand()

    print("All matrix tests passed!")
