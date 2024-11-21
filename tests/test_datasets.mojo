import testing

from mocrograd import datasets


fn test_datasets() raises:
    var dataset = datasets.DigitsData(10)

    testing.assert_equal(len(dataset.train_data), 8)
    testing.assert_equal(len(dataset.train_target), 8)
    testing.assert_equal(len(dataset.validation_data), 2)
    testing.assert_equal(len(dataset.validation_target), 2)

    var train_data_example = dataset.train_data[0]

    testing.assert_equal(train_data_example.rows, 64)
    testing.assert_equal(train_data_example.cols, 1)

    var validation_data_example = dataset.validation_data[0]

    testing.assert_equal(validation_data_example.rows, 64)
    testing.assert_equal(validation_data_example.cols, 1)

    var train_target_example = dataset.train_target[0]

    testing.assert_equal(train_target_example.rows, 1)
    testing.assert_equal(train_target_example.cols, 10)

    var validation_target_example = dataset.validation_target[0]

    testing.assert_equal(validation_target_example.rows, 1)
    testing.assert_equal(validation_target_example.cols, 10)

    testing.assert_true(train_data_example.data.max().item() <= 1)
    testing.assert_true(validation_data_example.data.max().item() <= 1)


fn run_tests() raises:
    test_datasets()

    print("All dataset tests passed!")
