import testing
from collections import Set

from mocrograd import tensor, grads, matrix


fn _create_matrix() -> matrix.Matrix:
    var data = matrix.Matrix(2, 3)
    data[0, 0] = 1.0
    data[0, 1] = 2.0
    data[0, 2] = 3.0
    data[1, 0] = 4.0
    data[1, 1] = 5.0
    data[1, 2] = 6.0
    return data


fn test_get_set() raises:
    var m = _create_matrix()

    testing.assert_equal(m[0, 0], 1.0)
    testing.assert_equal(m[0, 2], 3.0)
    testing.assert_equal(m[1, 1], 5.0)
    testing.assert_equal(m[1, 2], 6.0)


fn test_move() raises:
    var m1 = _create_matrix()
    testing.assert_equal(m1[1, 2], 6.0)

    var m2 = m1^
    testing.assert_equal(m2[1, 2], 6.0)
    # Cannot even access m1
    # testing.assert_equal(m1[1, 2], 6.0)


fn test_copy() raises:
    var m1 = _create_matrix()
    testing.assert_equal(m1[1, 2], 6.0)

    var m2 = m1
    testing.assert_equal(m1[1, 2], 6.0)
    testing.assert_equal(m2[1, 2], 6.0)


fn test_rand() raises:
    var t = matrix.Matrix.rand(3, 3)

    testing.assert_not_equal(t[0, 0], 0.0)
    testing.assert_not_equal(t[0, 1], 0.0)
    testing.assert_not_equal(t[2, 2], 0.0)


fn test_hash() raises:
    var m = matrix.Matrix.rand(3, 3)
    var s = Set(m)

    h = hash(m)
    testing.assert_not_equal(h, 1)

    testing.assert_equal(len(s), 1)

    s.add(m)

    testing.assert_equal(len(s), 1)


############################
# Tests for Math Functions #
############################


fn test_log() raises:
    var m = matrix.Matrix(1, 3)
    m[0, 0] = 0.0
    m[0, 1] = 1.0
    m[0, 2] = 10.0

    var log = m.log()

    testing.assert_equal(log[0, 0], Float32.MIN)
    testing.assert_equal(log[0, 1], 0.0)
    testing.assert_almost_equal(log[0, 2], 2.3025, atol=1e-4)


fn test_exp() raises:
    var m = matrix.Matrix(1, 3)
    m[0, 0] = 0.0
    m[0, 1] = 1.0
    m[0, 2] = 10.0

    var exp = m.exp()

    testing.assert_equal(exp[0, 0], 1.0)
    testing.assert_almost_equal(exp[0, 1], 2.7182, atol=1e-4)
    testing.assert_almost_equal(exp[0, 2], 22026.465, atol=1e-4)


fn test_max() raises:
    var m = matrix.Matrix(1, 3)
    m[0, 0] = 0.0
    m[0, 1] = -1.0
    m[0, 2] = 1.0

    var max = m.max()

    testing.assert_equal(max.item(), 1.0)


fn test_argmax() raises:
    var m = matrix.Matrix(1, 3)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0

    var max_index = m.argmax()

    testing.assert_equal(max_index, 2)


fn test_argmax_row() raises:
    var m = matrix.Matrix(3, 1)
    m[0, 0] = 1.0
    m[1, 0] = 6.0
    m[2, 0] = 3.0

    var max_index = m.argmax()

    testing.assert_equal(max_index, 1)


fn run_tests() raises:
    test_get_set()
    test_move()
    test_copy()
    test_rand()
    test_hash()
    # Tests for Math Functions
    test_log()
    test_exp()
    test_max()
    test_argmax()
    test_argmax_row()

    print("All matrix tests passed!")
