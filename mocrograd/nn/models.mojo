from collections import Dict

from mocrograd import tensor

from . import modules


struct MLP(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=2,
            out_features=16,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=16,
            out_features=16,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=16,
            out_features=1,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        return out

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigits(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=64,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=64,
            out_features=32,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=32,
            out_features=10,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigitsBigger(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=512,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=512,
            out_features=256,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=256,
            out_features=10,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^
