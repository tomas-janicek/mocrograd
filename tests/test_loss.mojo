import testing
from collections import Dict

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


fn test_cross_entropy() raises:
    var input = tensor.Tensor(10, 1)
    input[1, 0] = -1.0
    input[2, 0] = -7.0
    input[3, 0] = -2.0
    var target = tensor.Tensor(1, 10)
    target[0, 2] = 1.0

    var inputs = List(input)
    var targets = List(target)
    var cross_entropy = nn.cross_entropy_loss(
        inputs, targets, parameters_dict=Dict[String, List[tensor.Tensor]]()
    )

    testing.assert_equal(cross_entropy.item(), 7.0)


fn test_cross_entropy_bad() raises:
    var input = tensor.Tensor(10, 1)
    input[1, 0] = -1.0
    input[2, 0] = -7.0
    input[3, 0] = -2.0
    var target = tensor.Tensor(1, 10)
    target[0, 1] = 1.0

    var inputs = List(input)
    var targets = List(target)
    var cross_entropy = nn.cross_entropy_loss(
        inputs, targets, parameters_dict=Dict[String, List[tensor.Tensor]]()
    )

    testing.assert_equal(cross_entropy.item(), 1.0)


fn test_cross_entropy_multiple() raises:
    var input1 = tensor.Tensor(10, 1)
    input1[1, 0] = -1.0
    input1[2, 0] = -7.0
    input1[3, 0] = -2.0
    var input2 = tensor.Tensor(10, 1)
    input2[1, 0] = -3.0
    input2[2, 0] = -7.0
    input2[3, 0] = -2.0
    var target1 = tensor.Tensor(1, 10)
    target1[0, 2] = 1.0
    var target2 = tensor.Tensor(1, 10)
    target2[0, 1] = 1.0

    var inputs = List(input1, input2)
    var targets = List(target1, target2)
    var cross_entropy = nn.cross_entropy_loss(
        inputs, targets, parameters_dict=Dict[String, List[tensor.Tensor]]()
    )

    testing.assert_equal(cross_entropy.item(), 5.0)


fn run_tests() raises:
    test_accuracy()
    test_accuracy_bad()
    test_accuracy_half()
    test_cross_entropy()
    test_cross_entropy_bad()
    test_cross_entropy_multiple()

    print("All loss tests passed!")
