import tests


fn main() raises:
    tests.test_matrix.run_tests()
    tests.test_tensor.run_tests()
    tests.test_tensor_backward.run_tests()
    tests.test_loss.run_tests()
    tests.test_datasets.run_tests()

    print("All tests passed!")
