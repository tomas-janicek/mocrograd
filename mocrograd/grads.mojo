from mocrograd import tensor


trait Backward(Copyable, Movable):
    fn backward(self, out: tensor.Tensor) raises -> None:
        ...


struct NoneBackward(Backward):
    fn __init__(inout self):
        ...

    fn backward(self, out: tensor.Tensor) raises -> None:
        ...

    fn __copyinit__(inout self, other: Self):
        ...

    fn __moveinit__(inout self, owned existing: Self):
        ...


struct MatnulBackward[
    a_rows: Int,
    a_cols: Int,
    a_backward_t: Backward,
    b_rows: Int,
    b_cols: Int,
    b_backward_t: Backward,
](Backward):
    var a: tensor.Tensor[a_rows, a_cols, a_backward_t]
    var b: tensor.Tensor[b_rows, b_cols, b_backward_t]

    fn __init__(
        inout self,
        a: tensor.Tensor[a_rows, a_cols, a_backward_t],
        b: tensor.Tensor[b_rows, b_cols, b_backward_t],
    ):
        self.a = a
        self.b = b

    fn backward(self, out: tensor.Tensor) raises -> None:
        if not out.grad:
            raise "MissingGradError"
        if not self.a.grad and not self.b.grad:
            raise "MissingGradError"

        for m in range(self.a.rows):
            for k in range(self.a.cols):
                for n in range(self.b.cols):
                    if self.a.grad:
                        a_grad = self.a.grad.value()
                        a_grad_value = a_grad[m, k]
                        new_grad_value = self.b[k, n] * out.grad.value()[m, n]
                        a_grad.store(m, k, a_grad_value + new_grad_value)
                    if self.b.grad:
                        b_grad = self.b.grad.value()
                        b_grad_value = b_grad[m, k]
                        new_grad_value = self.a[m, k] * out.grad.value()[m, n]
                        b_grad.store(k, n, b_grad_value + new_grad_value)

    fn __copyinit__(inout self, existing: Self):
        self.a = existing.a
        self.b = existing.b

    fn __moveinit__(inout self, owned existing: Self):
        self.a = existing.a^
        self.b = existing.b^


struct SumBackward[
    rows: Int,
    cols: Int,
    backward_t: Backward,
](Backward):
    var a: tensor.Tensor[rows, cols, backward_t]

    fn __init__(
        inout self,
        a: tensor.Tensor[rows, cols, backward_t],
    ):
        self.a = a

    fn backward(self, out: tensor.Tensor) raises -> None:
        if not out.grad or self.a.grad:
            raise "MissingGradError"

        out_grad = out.grad.value()[0, 0]
        for row in range(self.a.rows):
            for col in range(self.a.cols):
                a_grad = self.a.grad.value()
                a_grad_value = a_grad[row, col]
                a_grad.store(row, col, a_grad_value + out_grad)

    fn __copyinit__(inout self, existing: Self):
        self.a = existing.a

    fn __moveinit__(inout self, owned existing: Self):
        self.a = existing.a^
