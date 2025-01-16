from collections import Optional, Set

from mocrograd import grads, utils, matrix


alias type = DType.float32


struct Tensor(Copyable, Movable, KeyElement):
    alias BackwardFunction = fn (
        out: matrix.Matrix,
        grad: Optional[matrix.Matrix],
        previousious: List[Tensor],
        grad_args: List[Float32],
    ) raises -> None

    var data: matrix.Matrix
    var rows: Int
    var cols: Int

    var requires_grad: Bool
    var grad: Optional[matrix.Matrix]
    var _op: StringLiteral
    var _previous: List[Tensor]
    var _grad_args: List[Float32]
    var _backward: Tensor.BackwardFunction

    # Create zeroed
    fn __init__(
        out self: Tensor,
        rows: Int,
        cols: Int,
        requires_grad: Bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.data = matrix.Matrix(self.rows, self.cols)

        self.requires_grad = requires_grad
        self._op = ""
        self._previous = List[Tensor]()
        self._grad_args = List[Float32]()
        self._backward = default_backward

        if self.requires_grad:
            self.grad = matrix.Matrix(self.rows, self.cols)
        else:
            self.grad = None

    # Create form data
    fn __init__(
        out self: Tensor,
        owned data: matrix.Matrix,
        requires_grad: Bool = False,
    ):
        self.rows = data.rows
        self.cols = data.cols
        self.data = data^

        self.requires_grad = requires_grad
        self._op = ""
        self._previous = List[Tensor]()
        self._grad_args = List[Float32]()
        self._backward = default_backward

        if self.requires_grad:
            self.grad = matrix.Matrix(self.rows, self.cols)
        else:
            self.grad = None

    # Create from scalar
    fn __init__(
        out self: Tensor,
        owned value: Float32,
        requires_grad: Bool = False,
    ):
        self.data = matrix.Matrix(value)
        self.rows = self.data.rows
        self.cols = self.data.cols

        self.requires_grad = requires_grad
        self._op = ""
        self._previous = List[Tensor]()
        self._grad_args = List[Float32]()
        self._backward = default_backward

        if self.requires_grad:
            self.grad = matrix.Matrix(self.rows, self.cols)
        else:
            self.grad = None

    # Create zeroed with grad information
    fn __init__(
        out self: Tensor,
        rows: Int,
        cols: Int,
        owned backward: Tensor.BackwardFunction,
        owned op: StringLiteral,
        owned previous: List[Tensor],
        owned grad_args: List[Float32],
        requires_grad: Bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.data = matrix.Matrix(self.rows, self.cols)

        self.requires_grad = requires_grad
        self._op = op
        self._previous = previous
        self._grad_args = grad_args
        self._backward = backward

        if self.requires_grad:
            self.grad = matrix.Matrix(self.rows, self.cols)
        else:
            self.grad = None

    # Create from data with grad information
    fn __init__(
        out self: Tensor,
        owned data: matrix.Matrix,
        owned backward: Tensor.BackwardFunction,
        owned op: StringLiteral,
        owned previous: List[Tensor],
        owned grad_args: List[Float32],
        requires_grad: Bool = False,
    ):
        self.rows = data.rows
        self.cols = data.cols
        self.data = data^

        self.requires_grad = requires_grad
        self._op = op
        self._previous = previous^
        self._grad_args = grad_args^
        self._backward = backward

        if self.requires_grad:
            self.grad = matrix.Matrix(self.rows, self.cols)
        else:
            self.grad = None

    @staticmethod
    fn rand(rows: Int, cols: Int, requires_grad: Bool = False) -> Tensor:
        var data = matrix.Matrix.rand(rows, cols)
        return Tensor(data, requires_grad=requires_grad)

    fn __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data
        self.requires_grad = existing.requires_grad
        self.grad = existing.grad
        self._op = existing._op
        self._previous = existing._previous
        self._grad_args = existing._grad_args
        self._backward = existing._backward

    fn __moveinit__(out self, owned existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data^
        self.requires_grad = existing.requires_grad
        self.grad = existing.grad^
        self._op = existing._op
        self._previous = existing._previous^
        self._grad_args = existing._grad_args^
        self._backward = existing._backward

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.data.load(y, x)

    fn __setitem__(mut self: Tensor, y: Int, x: Int, val: Scalar[type]):
        self.data.store(y, x, val)

    fn __hash__(self) -> UInt:
        return hash(self.data)

    fn __eq__(self, other: Self) -> Bool:
        return self.data == other.data

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn item(self) raises -> Scalar[type]:
        self._is_scalar()

        return self[0, 0]

    fn __matmul__(self, other: Tensor) raises -> Tensor:
        if self.cols != other.rows:
            raise "DimensionsError"

        var out = Tensor(
            data=self.data @ other.data,
            op="@",
            previous=List(self, other),
            grad_args=List[Float32](),
            backward=grads.matmul_backward,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        return out

    fn __add__(self, other: Tensor) raises -> Tensor:
        if (self.rows != other.rows) or (self.cols != other.cols):
            raise "DimensionsError"

        var out = Tensor(
            data=self.data + other.data,
            op="+",
            previous=List(self, other),
            grad_args=List[Float32](),
            backward=grads.addition_backward,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        return out

    fn __add__(self, other: Float32) raises -> Tensor:
        var out = Tensor(
            data=self.data + other,
            op="+",
            previous=List(self),
            grad_args=List[Float32](),
            backward=grads.addition_number_backward,
            requires_grad=self.requires_grad,
        )
        return out

    fn __radd__(self, other: Float32) raises -> Tensor:  # other + self
        return self + other

    fn __mul__(self, other: Float32) -> Tensor:
        out = Tensor(
            data=self.data * other,
            op="*",
            previous=List(self),
            grad_args=List(other),
            backward=grads.mul_backward,
            requires_grad=self.requires_grad,
        )
        return out

    fn __rmul__(self, other: Float32) -> Tensor:  # other * self
        return self * other

    fn __pow__(self, other: Float32) -> Tensor:
        out = Tensor(
            data=self.data**other,
            op="**",
            previous=List(self),
            grad_args=List(other),
            backward=grads.power_backward,
            requires_grad=self.requires_grad,
        )
        return out

    fn __neg__(self) -> Tensor:  # -self
        return self * -1.0

    fn __sub__(self, other: Tensor) raises -> Tensor:  # self - other
        return self + (-other)

    fn __sub__(self, other: Float32) raises -> Tensor:  # self - other
        return self + (-other)

    fn __rsub__(self, other: Float32) raises -> Tensor:  # other - self
        return other + (-self)

    fn __truediv__(self, other: Float32) -> Tensor:  # self / other
        return self * other**-1

    fn __rtruediv__(self, other: Float32) -> Tensor:  # other / self
        return other * self**-1

    fn sum(self) -> Tensor:
        var out = Tensor(
            data=self.data.sum(),
            op="sum",
            previous=List(self),
            grad_args=List[Float32](),
            backward=grads.sum_backward,
            requires_grad=self.requires_grad,
        )
        return out

    fn relu(self) -> Tensor:
        var out = Tensor(
            data=self.data.relu(),
            op="relu",
            previous=List(self),
            grad_args=List[Float32](),
            backward=grads.relu_backward,
            requires_grad=self.requires_grad,
        )
        return out

    fn log_softmax(self) raises -> Tensor:
        self._is_vector()

        var max_value = self.data.max()

        var stabilized_original = self.data - max_value.item()
        var exponentials = stabilized_original.exp()
        var sum_exponentials = exponentials.sum()

        var log_sum_exponentials = sum_exponentials.log()
        var log_probabilities_data = stabilized_original - log_sum_exponentials.item()

        var log_probabilities = Tensor(
            data=log_probabilities_data,
            op="log_softmax",
            previous=List(self),
            grad_args=List[Float32](),
            backward=grads.log_softmax_backward,
            requires_grad=self.requires_grad,
        )
        return log_probabilities

    fn backward(self) raises -> None:
        var backward_topology = List[Tensor]()
        var visited = Set[Tensor]()
        var queue = List[Tensor](self)

        while queue:
            var tensor = queue.pop()
            visited.add(tensor)

            if utils.all_in(items=tensor._previous, in_set=visited):
                backward_topology.append(tensor^)
            else:
                var previous = tensor._previous
                queue.append(tensor^)
                for prev_ref in previous:
                    if prev_ref[] not in visited:
                        queue.append(prev_ref[])

        self._set_grad_to_ones()
        # go one variable at a time and apply the chain rule to get its gradient
        for tensor_ref in reversed(backward_topology):
            var tensor = tensor_ref[]
            tensor._backward(
                out=tensor.data,
                grad=tensor.grad,
                previousious=tensor._previous,
                grad_args=tensor._grad_args,
            )

    fn _set_grad_to_ones(self) raises -> None:
        if not self.grad:
            raise "MissingGradError"

        for row in range(self.rows):
            for col in range(self.cols):
                self.grad.value().store(row, col, 1.0)

    fn _is_scalar(self) raises -> None:
        if self.rows != 1 or self.cols != 1:
            raise "DimensionsError"

    fn _is_vector(self) raises -> None:
        if self.cols != 1:
            raise "DimensionsError"

    fn __str__(self) raises -> String:
        var _str = String("Tensor(rows={}, cols={}, requires_grad={})").format(
            self.rows, self.cols, self.requires_grad
        )
        return _str

    fn print(self) -> None:
        var line = String("[")

        for row in range(self.rows):
            line += "["
            for col in range(self.cols - 1):
                line += Str(self[row, col]) + ", "
            line += Str(self[row, self.cols - 1]) + "],"

        line += "]"
        print(line)


fn default_backward(
    out: matrix.Matrix,
    grad: Optional[matrix.Matrix],
    previousious: List[Tensor],
    grad_args: List[Float32],
) raises -> None:
    return None
