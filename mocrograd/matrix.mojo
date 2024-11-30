import math
import random

from os import Atomic
from sys import info, simdwidthof
from memory import memset_zero, memcpy, UnsafePointer, ArcPointer
from algorithm import parallelize, vectorize
from utils import BlockingScopedLock, BlockingSpinLock

alias type = DType.float32

alias nelts = get_simd_width()
alias num_workers = 256


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
    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = ArcPointer(UnsafePointer[Scalar[type]].alloc(rows * cols))
        memset_zero(self.data[], rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(
        inout self,
        rows: Int,
        cols: Int,
        owned data: ArcPointer[UnsafePointer[Scalar[type]]],
    ):
        self.rows = rows
        self.cols = cols
        self.data = data^

    fn __init__(inout self, owned value: Float32):
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

    fn __setitem__(inout self: Matrix, row: Int, col: Int, val: Scalar[type]):
        self.store(row, col, val)

    fn load[nelts: Int = 1](self, row: Int, col: Int) -> SIMD[type, nelts]:
        return self.data[].load[width=nelts](row * self.cols + col)

    fn store[nelts: Int = 1](self, row: Int, col: Int, val: SIMD[type, nelts]):
        if row >= self.rows or col >= self.cols:
            print("IndexError")
        self.data[].store(row * self.cols + col, val)

    fn __copyinit__(inout self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data

    fn __moveinit__(inout self, owned existing: Self):
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

        parallelize[_matmul_row](out.rows, num_workers)

        return out

    fn __rmatmul__(self, other: Matrix) raises -> Matrix:  # other @ self
        return other @ self

    fn __add__(self, other: Matrix) raises -> Matrix:
        if (self.rows != other.rows) or (self.cols != other.cols):
            raise "DimensionsError"

        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _addition_row(row: Int):
            @parameter
            fn _addition[nelts: Int](col: Int):
                # out[row, col] = self[row, col] + other[row, col]
                out.store[nelts](
                    row,
                    col,
                    self.load[nelts](row, col) + other.load[nelts](row, col),
                )

            vectorize[_addition, nelts](size=out.cols)

        parallelize[_addition_row](out.rows, num_workers)

        return out

    fn __radd__(self, other: Matrix) raises -> Matrix:  # other + self
        return self + other

    fn __add__(self, other: Float32) raises -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _addition_row(row: Int):
            @parameter
            fn _addition[nelts: Int](col: Int):
                # out[row, col] = self[row, col] + other
                out.store[nelts](
                    row,
                    col,
                    self.load[nelts](row, col) + other,
                )

            vectorize[_addition, nelts](size=out.cols)

        parallelize[_addition_row](out.rows, num_workers)

        return out

    fn __radd__(self, other: Float32) raises -> Matrix:  # other + self
        return self + other

    fn __mul__(self, other: Float32) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _mul_row(row: Int):
            @parameter
            fn _mul[nelts: Int](col: Int):
                # out[row, col] = self[row, col] * other
                out.store[nelts](
                    row,
                    col,
                    self.load[nelts](row, col) * other,
                )

            vectorize[_mul, nelts](size=out.cols)

        parallelize[_mul_row](out.rows, num_workers)

        return out

    fn __rmul__(self, other: Float32) -> Matrix:  # other * self
        return self * other

    fn __pow__(self, other: Float32) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _pow_row(row: Int):
            @parameter
            fn _pow[nelts: Int](col: Int):
                # out[row, col] = self[row, col] ** other
                out.store[nelts](
                    row,
                    col,
                    self.load[nelts](row, col) ** other,
                )

            vectorize[_pow, nelts](size=out.cols)

        parallelize[_pow_row](out.rows, num_workers)

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
        var total_sum = Atomic(Float32(0.0))
        var num_workers = self.rows

        @parameter
        fn _sums_row(row: Int):
            for col in range(self.cols):
                _ = total_sum.fetch_add(self[row, col])

        parallelize[_sums_row](self.rows, num_workers)

        var out = Matrix(total_sum.load())
        return out

    fn max(self) -> Matrix:
        var max_value = Atomic(Float32.MIN)
        var num_workers = self.rows

        @parameter
        fn _max_row(row: Int):
            for col in range(self.cols):
                _ = max_value.max(self[row, col])

        parallelize[_max_row](self.rows, num_workers)

        var out = Matrix(max_value.load())
        return out

    fn relu(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _relu_row(row: Int):
            @parameter
            fn _relu[nelts: Int](col: Int):
                # out[row, col] = max(self[row, col], 0)
                out.store[nelts](
                    row,
                    col,
                    max(self.load[nelts](row, col), 0),
                )

            vectorize[_relu, nelts](size=out.cols)

        parallelize[_relu_row](out.rows, num_workers)

        return out

    fn exp(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _exp_row(row: Int):
            @parameter
            fn _exp[nelts: Int](col: Int):
                # out[row, col] = math.exp(self[row, col])
                out.store[nelts](
                    row,
                    col,
                    math.exp(self.load[nelts](row, col)),
                )

            vectorize[_exp, nelts](size=out.cols)

        parallelize[_exp_row](out.rows, num_workers)

        return out

    fn log(self) -> Matrix:
        var out = Matrix(rows=self.rows, cols=self.cols)

        @parameter
        fn _log_row(row: Int):
            @parameter
            fn _log[nelts: Int](col: Int):
                # out[row, col] = math.log(self[row, col])
                out.store[nelts](
                    row,
                    col,
                    math.log(self.load[nelts](row, col)),
                )

            vectorize[_log, nelts](size=out.cols)

        parallelize[_log_row](out.rows, num_workers)

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
