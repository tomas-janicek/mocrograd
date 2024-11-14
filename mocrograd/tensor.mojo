from memory import memset_zero, memcpy
from collections import Optional
from mocrograd import grads
from random import rand

alias type = DType.float32


struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroing all values
    fn __init__(inout self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load(y, x)

    fn __setitem__(
        inout self: Matrix[rows, cols], y: Int, x: Int, val: Scalar[type]
    ):
        self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[type, nelts]):
        self.data.store[width=nelts](y * self.cols + x, val)

    fn __copyinit__(inout self, existing: Self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memcpy(self.data, existing.data, rows * cols)

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        existing.data = UnsafePointer[Scalar[type]]()


struct Tensor[rows: Int, cols: Int, backward_t: grads.Backward](
    Copyable, Movable
):
    var data: Matrix[rows, cols]
    var requires_grad: Bool
    var grad: Optional[Matrix[rows, cols]]
    var _backward: backward_t

    fn __init__[
        backward_t: grads.Backward
    ](
        inout self: Tensor[rows, cols, backward_t],
        owned backward: backward_t,
        requires_grad: Bool = False,
    ):
        self.data = Matrix[rows, cols]()
        self.requires_grad = requires_grad
        self._backward = backward
        if self.requires_grad:
            self.grad = Matrix[rows, cols]()
        else:
            self.grad = None

    fn __init__(
        inout self: Tensor[rows, cols, grads.NoneBackward],
        requires_grad: Bool = False,
    ):
        self.data = Matrix[rows, cols]()
        self.requires_grad = requires_grad
        self._backward = grads.NoneBackward()

        if self.requires_grad:
            self.grad = Matrix[rows, cols]()
        else:
            self.grad = None

    fn __init__(
        inout self: Tensor[rows, cols, grads.NoneBackward],
        owned data: Matrix[rows, cols],
        requires_grad: Bool = False,
    ):
        self.data = data
        self.requires_grad = requires_grad
        self._backward = grads.NoneBackward()

        if self.requires_grad:
            self.grad = Matrix[rows, cols]()
        else:
            self.grad = None

    @staticmethod
    fn rand() -> Tensor[rows, cols, grads.NoneBackward]:
        var data = Matrix[rows, cols].rand()
        return Tensor[rows, cols, grads.NoneBackward](data, requires_grad=False)

    fn __copyinit__(inout self, existing: Self):
        self.requires_grad = existing.requires_grad
        self._backward = existing._backward
        self.data = existing.data
        self.grad = existing.grad

    fn __moveinit__(inout self, owned existing: Self):
        self.requires_grad = existing.requires_grad
        self._backward = existing._backward^
        self.data = existing.data^
        self.grad = existing.grad^

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.data.load(y, x)

    fn __setitem__(
        self: Tensor[rows, cols, backward_t], y: Int, x: Int, val: Scalar[type]
    ):
        self.data.store(y, x, val)

    fn item(self) raises -> Scalar[type]:
        self._is_scalar()

        return self[0, 0]

    fn _is_scalar(self) raises -> None:
        @parameter
        if rows != 1 or cols != 1:
            raise "DimensionsError"

    fn __matmul__(
        self, other: Tensor
    ) raises -> Tensor[
        rows,
        other.cols,
        grads.MatnulBackward[
            rows, cols, backward_t, other.rows, other.cols, other.backward_t
        ],
    ]:
        if self.cols != other.rows:
            raise "DimensionsError"

        var matmul_backward = grads.MatnulBackward[
            rows, cols, backward_t, other.rows, other.cols, other.backward_t
        ](self, other)

        out = Tensor[
            rows,
            other.cols,
            grads.MatnulBackward[
                rows, cols, backward_t, other.rows, other.cols, other.backward_t
            ],
        ](
            matmul_backward,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        for m in range(out.rows):
            for k in range(self.cols):
                for n in range(out.cols):
                    out[m, n] += self[m, k] * other[k, n]
        return out

    fn sum(
        self,
    ) -> Tensor[1, 1, grads.SumBackward[rows, cols, backward_t]]:
        var total_sum = Float32(0.0)
        for row in range(self.rows):
            for col in range(self.cols):
                total_sum += self[row, col]

        var sum_backward = grads.SumBackward[rows, cols, backward_t](self)
        out = Tensor[1, 1](
            backward=sum_backward, requires_grad=self.requires_grad
        )
        out[0, 0] = total_sum
        return out
