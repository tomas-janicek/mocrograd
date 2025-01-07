from collections import Dict

from mocrograd import tensor

from . import modules


struct MLP(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(out self) raises:
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

    fn __moveinit__(out self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigits(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(out self) raises:
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

    fn __moveinit__(out self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigitsBigger(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear
    var l4: modules.Linear

    fn __init__(out self) raises:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=8192,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=8192,
            out_features=4096,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=4096,
            out_features=2048,
            initialization="kaiming",
        )
        self.l4 = modules.Linear(
            in_features=2048,
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
        out = out.relu()
        out = self.l4.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()
        parameters["l4"] = self.l4.parameters()

        return parameters

    fn __moveinit__(out self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^
        self.l4 = existing.l4^


struct MLPDigitsLonger(modules.Module):
    var sequential_layers: List[modules.Linear]

    fn __init__(out self, n_layers: UInt) raises:
        if n_layers < 2:
            raise "BadLayersArgument"

        self.sequential_layers = List[modules.Linear]()
        var first = modules.Linear(
            in_features=8 * 8,
            out_features=256,
            initialization="kaiming",
        )
        self.sequential_layers.append(first^)
        for _ in range(n_layers - 2):
            lx = modules.Linear(
                in_features=256,
                out_features=256,
                initialization="kaiming",
            )
            self.sequential_layers.append(lx^)
        last = modules.Linear(
            in_features=256,
            out_features=10,
            initialization="kaiming",
        )
        self.sequential_layers.append(last^)

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        var out = input
        var i = 0
        for layer in self.sequential_layers:
            out = layer[].forward(out)
            if i == len(self.sequential_layers) - 1:
                out = out.log_softmax()
            else:
                out = out.relu()
            i += 1

        return out

    fn parameters(self) raises -> modules.ParametersDict:
        var i = 0
        var parameters = modules.ParametersDict()
        for layer in self.sequential_layers:
            var layer_key = String("l{}").format(i)
            parameters[layer_key] = layer[].parameters()

        return parameters

    fn __moveinit__(out self, owned existing: Self):
        self.sequential_layers = existing.sequential_layers^
