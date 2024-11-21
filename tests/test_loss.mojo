import testing

from mocrograd import nn, tensor


fn test_accuracy() raises:
    var input = tensor.Tensor(10, 1)
    input[2, 0] = 1.0
    var target = tensor.Tensor(1, 10)
    target[0, 2] = 1.0
    var inputs = List(input)
    var targets = List(target)
    var accuracy = nn.calculate_accuracy(inputs, targets)

    testing.assert_equal(accuracy, 1)


fn test_accuracy_bad() raises:
    var input = tensor.Tensor(10, 1)
    input[3, 0] = 1.0
    var target = tensor.Tensor(1, 10)
    target[0, 2] = 1.0
    var inputs = List(input)
    var targets = List(target)
    var accuracy = nn.calculate_accuracy(inputs, targets)

    testing.assert_equal(accuracy, 0)


fn test_accuracy_half() raises:
    var input = tensor.Tensor(10, 1)
    input[2, 0] = 0.5
    var target = tensor.Tensor(1, 10)
    target[0, 2] = 1.0
    var inputs = List(input)
    var targets = List(target)
    var accuracy = nn.calculate_accuracy(inputs, targets)

    testing.assert_equal(accuracy, 1)


fn run_tests() raises:
    test_accuracy()
    test_accuracy_bad()
    test_accuracy_half()

    print("All loss tests passed!")
