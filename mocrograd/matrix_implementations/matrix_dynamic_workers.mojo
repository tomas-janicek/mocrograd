import math
import random

from os import Atomic
from sys import info, simdwidthof
from memory import memset_zero, memcpy, UnsafePointer, ArcPointer
from algorithm import vectorize, parallelize
from utils import BlockingScopedLock, BlockingSpinLock

alias type = DType.float32

alias nelts = get_simd_width()


fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()


struct Matrix(Copyable, Movable, KeyElement):
    var data: ArcPointer[UnsafePointer[Scalar[type]]]
    var rows: Int
    var cols: Int

    # Initialize zeroing all values
    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = ArcPointer(UnsafePointer[Scalar[type]].alloc(rows * cols))
        memset_zero(self.data[], rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(
        out self,
        rows: Int,
        cols: Int,
        owned data: ArcPointer[UnsafePointer[Scalar[type]]],
    ):
        self.rows = rows
        self.cols = cols
        self.data = data^

    fn __init__(out self, owned value: Float32):
        self.rows = 1
        self.cols = 1
        self.data = ArcPointer(UnsafePointer[Scalar[type]].alloc(1))
        self.data[].init_pointee_move(value)

    # Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        random.rand(data, rows * cols)
        return Self(rows, cols, ArcPointer(data))

    @staticmethod
    fn randn(rows: Int, cols: Int) -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        random.randn(data, rows * cols)
        return Self(rows, cols, ArcPointer(data))

    fn item(self) raises -> Scalar[type]:
        self._is_scalar()

        return self[0, 0]

    fn __getitem__(self, row: Int, col: Int) -> Scalar[type]:
        return self.load(row, col)

    fn __setitem__(mut self: Matrix, row: Int, col: Int, val: Scalar[type]):
        self.store(row, col, val)

    fn load[nelts: Int = 1](self, row: Int, col: Int) -> SIMD[type, nelts]:
        return self.data[].load[width=nelts](row * self.cols + col)

    fn store[nelts: Int = 1](self, row: Int, col: Int, val: SIMD[type, nelts]):
        self.data[].store(row * self.cols + col, val)

    fn __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data

    fn __moveinit__(out self, owned existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data^

    fn __hash__(self) -> UInt:
        var unit_data = self.data[].bitcast[UInt8]()
        # We must multiply length of our data (rows * cols) by 4
        # because UInt8 is 4 times smaller that Float32.
        var matrix_hash = hash(unit_data, self.rows * self.cols * 4)
        return matrix_hash

    fn __eq__(self, other: Self) -> Bool:
        return hash(self) == hash(other)

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __matmul__(self, other: Matrix) raises -> Matrix:
        if self.cols != other.rows:
            raise "DimensionsError"

        var out = Matrix(
            rows=self.rows,
            cols=other.cols,
        )

        @parameter
        fn _matmul_row(m: Int):
            for k in range(self.cols):

                @parameter
                fn _matmul[nelts: Int](n: Int):
                    out.store[nelts](
                        m,
                        n,
                        out.load[nelts](m, n)
                        + self[m, k] * other.load[nelts](k, n),
                    )

                vectorize[_matmul, nelts](size=out.cols)

        parallelize[_matmul_row](out.rows, out.rows)

        return out

    fn __rmatmul__(self, other: Matrix) raises -> Matrix:  # other @ self
        return other @ self

    fn __add__(self, other: Matrix) raises -> Matrix:
        if (self.rows != other.rows) or (self.cols != other.cols):
            raise "DimensionsError"

        var out = Matrix(rows=self.rows, cols=self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] + other[row, col]
        return out

    fn __radd__(self, other: Matrix) raises -> Matrix:  # other + self
        return self + other

    fn __add__(self, other: Float32) raises -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] + other
        return out

    fn __radd__(self, other: Float32) raises -> Matrix:  # other + self
        return self + other

    fn __mul__(self, other: Float32) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] * other

        return out

    fn __rmul__(self, other: Float32) -> Matrix:  # other * self
        return self * other

    # TODO: Write tests to all this function on tensor side
    fn __pow__(self, other: Float32) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] ** other

        return out

    fn __neg__(self) -> Matrix:  # -self
        return self * -1.0

    fn __sub__(self, other: Matrix) raises -> Matrix:  # self - other
        return self + (-other)

    fn __sub__(self, other: Float32) raises -> Matrix:  # self - other
        return self + (-other)

    fn __rsub__(self, other: Matrix) raises -> Matrix:  # other - self
        return other + (-self)

    fn __rsub__(self, other: Float32) raises -> Matrix:  # other - self
        return other + (-self)

    fn __truediv__(self, other: Float32) -> Matrix:  # self / other
        return self * other**-1

    fn __rtruediv__(self, other: Float32) -> Matrix:  # other / self
        return other * self**-1

    fn sum(self) -> Matrix:
        var total_sum = Float32(0.0)
        for row in range(self.rows):
            for col in range(self.cols):
                total_sum += self[row, col]

        var out = Matrix(total_sum)
        return out

    fn max(self) -> Matrix:
        var max_value = Float32(0.0)
        for row in range(self.rows):
            for col in range(self.cols):
                max_value = max(max_value, self[row, col])

        var out = Matrix(max_value)
        return out

    fn relu(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = max(self[row, col], 0)

        return out

    fn exp(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = math.exp(self[row, col])

        return out

    fn log(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = math.log(self[row, col])

        return out

    fn argmax(self) raises -> UInt:
        self._is_one_dimensional()

        var max_index_row = 0
        var max_index_col = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if self[row, col] > self[max_index_row, max_index_col]:
                    max_index_row = row
                    max_index_col = col

        # Because self can be both vector (vertical array) and matrix (horizontal array),
        # we return the bigger index because the other one is always equal to one.
        return max_index_row if max_index_row > max_index_col else max_index_col

    fn _is_scalar(self) raises -> None:
        if self.rows != 1 or self.cols != 1:
            raise "DimensionsError"

    fn _is_one_dimensional(self) raises -> None:
        if self.cols != 1 and self.rows != 1:
            raise "DimensionsError"

    fn __str__(self) raises -> String:
        var _str = String("Matrix(rows={}, cols={}, requires_grad={})").format(
            self.rows, self.cols
        )
        return _str
