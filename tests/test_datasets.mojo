import testing

from mocrograd import datasets, tensor


struct FakeDataset(datasets.Dataset):
    var train_data: List[tensor.Tensor]
    var train_target: List[tensor.Tensor]
    var validation_data: List[tensor.Tensor]
    var validation_target: List[tensor.Tensor]

    fn __init__(
        out self,
        train_len: UInt,
        validation_len: UInt,
        data_rows: UInt = 3,
        data_cols: UInt = 1,
        target_rows: UInt = 1,
        target_cols: UInt = 3,
    ):
        self.train_data = List[tensor.Tensor]()
        self.train_target = List[tensor.Tensor]()
        self.validation_data = List[tensor.Tensor]()
        self.validation_target = List[tensor.Tensor]()

        for _ in range(train_len):
            self.train_data.append(tensor.Tensor.rand(data_rows, data_cols))
            self.train_target.append(
                tensor.Tensor.rand(target_rows, target_cols)
            )

        for _ in range(validation_len):
            self.validation_data.append(
                tensor.Tensor.rand(data_rows, data_cols)
            )
            self.validation_target.append(
                tensor.Tensor.rand(target_rows, target_cols)
            )

    fn get_train_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        return self.train_data, self.train_target

    fn get_validation_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        return self.validation_data, self.validation_target

    fn __moveinit__(out self, owned existing: Self):
        self.train_data = existing.train_data^
        self.train_target = existing.train_target^
        self.validation_data = existing.validation_data^
        self.validation_target = existing.validation_target^


fn test_digits_datasets() raises:
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


fn test_dataloader_iterator_len() raises:
    var data = List(
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
    )
    var target = List(
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
    )
    var it = datasets.DataloaderIterator(data, target, 2)

    testing.assert_equal(len(it), 2)


fn test_dataloader_iterator_len_uneven() raises:
    var data = List(
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
    )
    var target = List(
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
    )
    var it = datasets.DataloaderIterator(data, target, 2)

    testing.assert_equal(len(it), 3)


fn test_dataloader_iteration() raises:
    var data = List(
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
    )
    var target = List(
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
    )
    var it = datasets.DataloaderIterator(data, target, 2)

    var iterations = 0
    for data_target in it:
        data, target = data_target
        testing.assert_equal(len(data), 2)
        testing.assert_equal(len(target), 2)
        iterations += 1

    testing.assert_equal(iterations, 2)


fn test_dataloader_iteration_uneven() raises:
    var data = List(
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
        tensor.Tensor.rand(3, 1),
    )
    var target = List(
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
        tensor.Tensor.rand(1, 3),
    )
    var it = datasets.DataloaderIterator(data, target, 2)

    var iterations = 0
    for data_target in it:
        data, target = data_target
        if iterations != 2:
            testing.assert_equal(len(data), 2)
            testing.assert_equal(len(target), 2)
        else:
            testing.assert_equal(len(data), 1)
            testing.assert_equal(len(target), 1)
        iterations += 1

    testing.assert_equal(iterations, 3)


fn test_data_loader_len() raises:
    var dataset = FakeDataset(train_len=20, validation_len=10)
    var dl = datasets.Dataloader(dataset^, batch_size=5)

    testing.assert_equal(len(dl.get_train_dataloader()), 4)
    testing.assert_equal(len(dl.get_validation_dataloader()), 2)


fn run_tests() raises:
    test_digits_datasets()
    test_dataloader_iterator_len()
    test_dataloader_iterator_len_uneven()
    test_dataloader_iteration()
    test_dataloader_iteration_uneven()
    test_data_loader_len()

    print("All dataset tests passed!")
