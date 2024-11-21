from python import Python

from mocrograd import matrix, tensor


fn load_digits(
    length: UInt,
) raises -> Tuple[List[matrix.Matrix], List[matrix.Matrix]]:
    var scikit_learn = Python.import_module("sklearn.datasets")
    var digits_dataset = scikit_learn.load_digits()
    var raw_data = digits_dataset.data
    var raw_target = digits_dataset.target

    var data = List[matrix.Matrix]()
    var target = List[matrix.Matrix]()

    for digits in raw_data:
        var m_data = matrix.Matrix(64, 1)
        var row = 0
        for digit in digits:
            var d = digit.to_float64().cast[DType.float32]()
            # We want to normalize the data into range between 0 and 1.
            m_data[row, 0] = d / 16
            row += 1

        data.append(m_data)

    for digit in raw_target:
        var d = digit.to_float64().cast[DType.uint32]()
        var m_data = _one_hot_encode(int(d), n_classes=10)

        target.append(m_data)

    length_slice = slice(length)
    return data[length_slice], target[length_slice]


fn _one_hot_encode(value: UInt, n_classes: UInt) -> matrix.Matrix:
    t = matrix.Matrix(rows=1, cols=n_classes)
    t[0, value] = 1.0
    return t
