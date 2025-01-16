from python import Python
from collections import Dict

from mocrograd import tensor, data

alias DataTuple = Tuple[List[tensor.Tensor], List[tensor.Tensor]]


trait Dataset(Movable):
    fn get_train_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        ...

    fn get_validation_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        ...

    fn __moveinit__(out self, owned existing: Self):
        ...


struct DataloaderIterator(Sized, Copyable):
    var data: List[tensor.Tensor]
    var target: List[tensor.Tensor]
    var current_state: UInt
    var batch_size: UInt
    var length: UInt
    var iterations: UInt

    fn __init__(
        out self,
        owned data: List[tensor.Tensor],
        owned target: List[tensor.Tensor],
        batch_size: UInt,
    ):
        self.data = data^
        self.target = target^
        self.current_state = 0
        self.length = len(self.data)
        self.batch_size = batch_size
        self.iterations = (self.length + self.batch_size - 1) // self.batch_size

    fn __iter__(self) -> Self:
        return self

    fn __len__(self) -> Int:
        return self.iterations

    fn __next__(mut self) -> DataTuple:
        var start = self.current_state
        var end = start + self.batch_size
        if end > self.length:
            end = self.length

        self.current_state += self.batch_size
        self.iterations -= 1

        var data = DataTuple(
            self.data[slice(start, end)], self.target[slice(start, end)]
        )
        return data^

    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.target = existing.target
        self.current_state = existing.current_state
        self.length = existing.length
        self.batch_size = existing.batch_size
        self.iterations = existing.iterations


struct Dataloader[DatasetT: Dataset]:
    var batch_size: UInt
    var dataset: DatasetT
    var train_data: List[tensor.Tensor]
    var train_target: List[tensor.Tensor]
    var validation_data: List[tensor.Tensor]
    var validation_target: List[tensor.Tensor]
    var train_length: UInt
    var validation_length: UInt

    fn __init__(out self, owned dataset: DatasetT, batch_size: UInt):
        self.batch_size = batch_size
        self.dataset = dataset^
        self.train_data, self.train_target = self.dataset.get_train_data()
        self.validation_data, self.validation_target = (
            self.dataset.get_validation_data()
        )

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)

    fn get_train_dataloader(
        self,
    ) -> DataloaderIterator:
        return DataloaderIterator(
            data=self.train_data,
            target=self.train_target,
            batch_size=self.batch_size,
        )

    fn get_validation_dataloader(
        self,
    ) -> DataloaderIterator:
        return DataloaderIterator(
            data=self.validation_data,
            target=self.validation_target,
            batch_size=self.batch_size,
        )


struct DigitsData(Dataset):
    var length: UInt
    var train_data: List[tensor.Tensor]
    var validation_data: List[tensor.Tensor]
    var train_target: List[tensor.Tensor]
    var validation_target: List[tensor.Tensor]

    fn __init__(out self, length: UInt) raises:
        self.length = length
        matrix_data, matrix_target = data.load_digits(length)

        train_data, validation_data, train_target, validation_target = (
            train_validation_split(
                matrix_data, matrix_target, validation_size=0.2
            )
        )

        self.train_data = _create_tensors(train_data)
        self.validation_data = _create_tensors(validation_data)
        self.train_target = _create_tensors(train_target)
        self.validation_target = _create_tensors(validation_target)

    fn get_train_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        return self.train_data, self.train_target

    fn get_validation_data(
        self,
    ) -> Tuple[List[tensor.Tensor], List[tensor.Tensor]]:
        return self.validation_data, self.validation_target

    fn __moveinit__(out self, owned existing: Self):
        self.length = existing.length
        self.train_data = existing.train_data
        self.validation_data = existing.validation_data
        self.train_target = existing.train_target
        self.validation_target = existing.validation_target


fn train_validation_split(
    data: List[matrix.Matrix],
    target: List[matrix.Matrix],
    validation_size: Float16,
) raises -> Tuple[
    List[matrix.Matrix],
    List[matrix.Matrix],
    List[matrix.Matrix],
    List[matrix.Matrix],
]:
    if len(data) != len(target):
        raise "InvalidDataTargetPair"
    var validation_len = Int(len(data) * validation_size)
    var train_length = len(data) - validation_len

    var train_slice = slice(train_length)
    var validation_slice = slice(train_length, train_length + validation_len)
    return (
        data[train_slice],
        data[validation_slice],
        target[train_slice],
        target[validation_slice],
    )


fn _create_tensors(
    read matrices: List[matrix.Matrix],
) -> List[tensor.Tensor]:
    var tensors = List[tensor.Tensor]()
    for matrix in matrices:
        var new_tensor = tensor.Tensor(matrix[])
        tensors.append(new_tensor)
    return tensors
